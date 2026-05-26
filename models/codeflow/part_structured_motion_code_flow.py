"""Part-structured frame-token CodeFlow.

This is the canonical PS-CF path: one DiT token per RVQ frame, six grouped
part-specific input/output paths, and terminal projection tied to the frozen
part codebooks.
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_blocks import FinalLayer, FrameMotionTextDiT, TimestepEmbedder
from .kv_vq import PartVQTokenizer
from .motion_code_flow import MotionCodeFlow, MotionCodeFlowConfig, lengths_to_mask, sample_timesteps
from .text_encoder import FrozenCLIPTextEncoder, TextCondition


class PartStructuredMotionCodeFlow(MotionCodeFlow):
    """Rectified flow over structured six-part RVQ frame embeddings."""

    def __init__(self, config: MotionCodeFlowConfig) -> None:
        nn.Module.__init__(self)
        if config.representation != "part_structured":
            raise ValueError("PartStructuredMotionCodeFlow requires representation='part_structured'")
        if config.coupling_mode != "frame_grouped":
            raise ValueError("PartStructuredMotionCodeFlow requires coupling_mode='frame_grouped'")
        if config.time_patch != 1:
            raise ValueError("PartStructuredMotionCodeFlow uses time_patch=1")
        if config.use_self_condition:
            raise ValueError("PartStructuredMotionCodeFlow canonical path disables self-conditioning")
        if float(config.clean_loss_weight) != 0.0:
            raise ValueError("PartStructuredMotionCodeFlow canonical objective uses clean_loss_weight=0")
        if config.hidden_size != config.num_parts * config.code_dim:
            raise ValueError(
                "PartStructuredMotionCodeFlow does not compress part channels: "
                f"hidden_size must be num_parts*code_dim={config.num_parts * config.code_dim}, "
                f"got {config.hidden_size}"
            )
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f"hidden_size {config.hidden_size} must be divisible by num_heads={config.num_heads}")
        if config.terminal_mode not in {"nearest", "tied_logits", "learned_head"}:
            raise ValueError(f"Unsupported terminal_mode: {config.terminal_mode}")
        if config.latent_norm_mode not in {"none", "codebook"}:
            raise ValueError(f"Unsupported latent_norm_mode: {config.latent_norm_mode}")
        if float(config.latent_offset) != 0.0:
            raise ValueError("PartStructuredMotionCodeFlow uses latent_offset=0 to preserve raw codebook metric")
        if config.sampling_schedule not in {"uniform", "logit_normal"}:
            raise ValueError(f"Unsupported sampling_schedule: {config.sampling_schedule}")
        if config.sampling_method not in {"ode", "sde"}:
            raise ValueError(f"Unsupported sampling_method: {config.sampling_method}")
        if config.decode_mode not in {"nearest", "ids", "continuous"}:
            raise ValueError(f"Unsupported decode_mode: {config.decode_mode}")
        if config.terminal_tau_mode not in {"fixed", "codebook_nn"}:
            raise ValueError(f"Unsupported terminal_tau_mode: {config.terminal_tau_mode}")
        self.config = config

        self.tokenizer = PartVQTokenizer(
            kv_root=config.kv_root,
            checkpoint_path=config.vq_checkpoint,
            partition_path=config.vq_partition,
        )
        if self.tokenizer.num_parts != config.num_parts:
            raise ValueError(f"Config num_parts={config.num_parts}, tokenizer has {self.tokenizer.num_parts}")
        if self.tokenizer.num_codes != config.num_codes:
            raise ValueError(f"Config num_codes={config.num_codes}, tokenizer has {self.tokenizer.num_codes}")
        if self.tokenizer.code_dim != config.code_dim:
            raise ValueError(f"Config code_dim={config.code_dim}, tokenizer has {self.tokenizer.code_dim}")
        self._init_latent_stats()
        self._init_terminal_tau()

        self.text_encoder = FrozenCLIPTextEncoder(
            clip_version=config.clip_version,
            clip_path=config.clip_path,
            kv_root=config.kv_root,
        )

        part_dim = config.code_dim
        self.part_input_norms = nn.ModuleList([
            nn.LayerNorm(config.code_dim, elementwise_affine=True, eps=1e-6)
            for _ in range(config.num_parts)
        ])
        self.part_inputs = nn.ModuleList([
            nn.Linear(config.code_dim, part_dim)
            for _ in range(config.num_parts)
        ])

        self.timestep_embed = TimestepEmbedder(config.hidden_size)
        self.text_token_proj = nn.Linear(self.text_encoder.width, config.hidden_size)
        self.text_pooled_proj = nn.Sequential(
            nn.Linear(self.text_encoder.output_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        head_dim = config.hidden_size // config.num_heads
        self.backbone = FrameMotionTextDiT(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            depth_double=config.depth_double,
            depth_single=config.depth_single,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            rope_axes_dims=[head_dim],
        )
        self.part_outputs = nn.ModuleList([
            FinalLayer(config.hidden_size, config.code_dim)
            for _ in range(config.num_parts)
        ])

        if config.terminal_mode == "learned_head":
            self.learned_heads = nn.ModuleList([
                nn.Linear(config.code_dim, config.num_codes)
                for _ in range(config.num_parts)
            ])
        else:
            self.learned_heads = None

    @property
    def holder_output(self):
        # Kept as a compatibility shim for older training health checks.
        return self.part_outputs[0]

    @property
    def core_output_weight(self) -> torch.Tensor:
        return self.part_outputs[0].linear.weight

    def _init_terminal_tau(self) -> None:
        cfg = self.config
        if cfg.terminal_tau_mode == "codebook_nn":
            values: List[torch.Tensor] = []
            for part_idx in range(cfg.num_parts):
                codebook = self.tokenizer.codebooks[part_idx].float()
                dist_sq = torch.cdist(codebook, codebook, p=2.0).square()
                dist_sq.fill_diagonal_(float("inf"))
                nearest = dist_sq.min(dim=1).values
                tau = torch.median(nearest[torch.isfinite(nearest)])
                values.append(tau.clamp_min(float(cfg.terminal_tau_floor)))
            tau_parts = torch.stack(values).float()
        else:
            tau_parts = torch.full(
                (cfg.num_parts,),
                max(float(cfg.terminal_tau), float(cfg.terminal_tau_floor)),
                dtype=torch.float32,
            )
        self.register_buffer("terminal_tau_parts", tau_parts, persistent=False)

    def _text_condition(
        self,
        texts: Iterable[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> TextCondition:
        cond = self.text_encoder(texts, drop_prob=drop_prob, force_drop=force_drop)
        return TextCondition(
            pooled=self.text_pooled_proj(cond.pooled),
            tokens=self.text_token_proj(cond.tokens),
            padding_mask=cond.padding_mask,
        )

    def _pack_motion(
        self,
        x: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        bsz, latent_len, num_parts, dim = x.shape
        if num_parts != cfg.num_parts or dim != cfg.code_dim:
            raise RuntimeError(
                f"Expected motion latent [B,T,{cfg.num_parts},{cfg.code_dim}], got {tuple(x.shape)}"
            )
        motion_valid = lengths_to_mask(token_lengths, latent_len)
        part_chunks = []
        for part_idx in range(cfg.num_parts):
            part_x = self.part_input_norms[part_idx](x[:, :, part_idx])
            part_chunks.append(self.part_inputs[part_idx](part_x))
        tokens = torch.cat(part_chunks, dim=-1)
        time_ids = torch.arange(latent_len, device=x.device, dtype=torch.float32)
        pos = time_ids.view(1, latent_len, 1).expand(bsz, -1, -1)
        return tokens, motion_valid, pos

    def forward(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None,
        text_drop_prob: float = 0.0,
        force_text_drop: bool = False,
    ) -> torch.Tensor:
        del x_self_cond
        cfg = self.config
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(z.shape[0])
        timesteps = timesteps.to(device=z.device, dtype=z.dtype)
        token_lengths = token_lengths.to(z.device).long().clamp(min=1, max=z.shape[1])

        text_cond = self._text_condition(texts, drop_prob=text_drop_prob, force_drop=force_text_drop)
        motion_tokens, motion_valid, motion_pos = self._pack_motion(z, token_lengths)
        cond = self.timestep_embed(timesteps.float()) + text_cond.pooled
        hidden = self.backbone(
            motion=motion_tokens,
            text=text_cond.tokens,
            cond=cond,
            motion_valid=motion_valid,
            text_padding_mask=text_cond.padding_mask,
            motion_pos_ids=motion_pos,
        )
        parts = [head(hidden, cond) for head in self.part_outputs]
        pred = torch.stack(parts, dim=2)
        valid = lengths_to_mask(token_lengths, z.shape[1]).to(pred.dtype)
        return pred * valid[:, :, None, None]

    def terminal_logits(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        mode = mode or self.config.terminal_mode
        if mode in {"nearest", "tied_logits"}:
            return self.tokenizer.codebook_tied_logits(clean_pred, tau=self.terminal_tau_parts)
        if mode == "learned_head":
            if self.learned_heads is None:
                raise RuntimeError("learned_head terminal mode was not initialized")
            logits = []
            for part_idx, head in enumerate(self.learned_heads):
                logits.append(head(clean_pred[:, :, part_idx]))
            return torch.stack(logits, dim=2)
        raise ValueError(f"Unknown terminal mode: {mode}")

    def compute_losses(
        self,
        target_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        include_geometry_metrics: bool = False,
        geometry_severe_quantile: float = 0.75,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        allow_internal_self_condition: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del x_self_cond, allow_internal_self_condition
        cfg = self.config
        bsz, latent_len, num_parts, _ = target_embeddings.shape
        token_lengths = token_lengths.to(target_embeddings.device).long().clamp(min=1, max=latent_len)
        valid = lengths_to_mask(token_lengths, latent_len)
        valid_parts = valid[:, :, None].expand(bsz, latent_len, num_parts)
        valid_float = valid_parts.to(target_embeddings.dtype)
        target_model = self.raw_to_model_latent(target_embeddings)

        if noise is None:
            noise = torch.randn_like(target_model) * cfg.noise_scale
        else:
            noise = noise.to(device=target_model.device, dtype=target_model.dtype)
        if timesteps is None:
            t = sample_timesteps(
                bsz,
                target_embeddings.device,
                cfg.time_schedule,
                cfg.denoiser_p_mean,
                cfg.denoiser_p_std,
            ).to(target_embeddings.dtype)
        else:
            t = timesteps.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
            if t.ndim == 0:
                t = t.expand(bsz)
        t_view = t[:, None, None, None]
        z_t = t_view * target_model + (1.0 - t_view) * noise
        velocity_target = target_model - noise
        z_t = z_t * valid_float[:, :, :, None]

        velocity_pred = self.forward(
            z_t,
            t,
            texts,
            token_lengths,
            x_self_cond=None,
            text_drop_prob=cfg.cond_drop_prob,
        )
        velocity_pred_f = velocity_pred.float()
        velocity_target_f = velocity_target.float()
        valid_float_f = valid_float.float()
        z_t_f = z_t.float()
        t_f = t.float()

        per_part_flow = (velocity_pred_f - velocity_target_f).square().mean(dim=-1)
        flow_loss = (per_part_flow * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)

        clean_pred = self.predict_clean_from_velocity(z_t_f, t_f, velocity_pred_f)
        clean_loss = flow_loss.new_zeros(())
        clean_pred_raw = self.model_to_raw_latent(clean_pred)

        terminal_loss = flow_loss.new_zeros(())
        code_weight = (
            (t_f >= float(cfg.code_ce_t_min))
            & (t_f <= float(cfg.code_ce_t_max))
        ).to(valid_float_f.dtype) * t_f.clamp_min(0.0).pow(float(cfg.code_ce_gamma))
        if cfg.terminal_mode in {"tied_logits", "learned_head"} and cfg.terminal_loss_weight > 0.0:
            with torch.cuda.amp.autocast(enabled=False):
                logits = self.terminal_logits(clean_pred_raw.float()).float()
                ce = F.cross_entropy(
                    logits.reshape(-1, cfg.num_codes),
                    target_ids.reshape(-1).long(),
                    reduction="none",
                ).view(bsz, latent_len, num_parts)
                if cfg.code_ce_normalize:
                    ce = ce / math.log(float(cfg.num_codes))
                ce = ce * code_weight[:, None, None]
            terminal_loss = (ce * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)

        with torch.no_grad():
            pred_ids = self.terminal_ids(clean_pred_raw)
            acc = ((pred_ids == target_ids.long()) & valid_parts).sum().float() / valid_parts.sum().float().clamp_min(1.0)
            nn_ids = self.tokenizer.nearest_ids(clean_pred_raw)
            nn_acc = ((nn_ids == target_ids.long()) & valid_parts).sum().float() / valid_parts.sum().float().clamp_min(1.0)

        total = cfg.flow_loss_weight * flow_loss + cfg.terminal_loss_weight * terminal_loss
        out = {
            "loss": total,
            "flow_loss": flow_loss,
            "terminal_loss": terminal_loss,
            "clean_loss": clean_loss,
            "token_acc": acc,
            "nearest_acc": nn_acc,
            "code_ce_weight": code_weight.mean(),
        }
        if include_geometry_metrics:
            with torch.no_grad():
                code_dist, rank_pct = self.tokenizer.code_id_distances(target_ids.long(), pred_ids.long())
                valid_bool = valid_parts.bool()
                wrong = (pred_ids != target_ids.long()) & valid_bool
                severe = wrong & (rank_pct >= float(geometry_severe_quantile))
                valid_count = valid_bool.sum().float().clamp_min(1.0)
                wrong_count = wrong.sum().float()
                wrong_denom = wrong_count.clamp_min(1.0)
                out.update({
                    "geom_code_dist": (code_dist * valid_bool.to(code_dist.dtype)).sum() / valid_count,
                    "geom_rank_pct": (rank_pct * valid_bool.to(rank_pct.dtype)).sum() / valid_count,
                    "geom_wrong_code_dist": (code_dist * wrong.to(code_dist.dtype)).sum() / wrong_denom,
                    "geom_wrong_rank_pct": (rank_pct * wrong.to(rank_pct.dtype)).sum() / wrong_denom,
                    "geom_wrong_rate": wrong_count / valid_count,
                    "geom_severe_rate": severe.sum().float() / valid_count,
                    "geom_wrong_severe_frac": severe.sum().float() / wrong_denom,
                })
        return out
