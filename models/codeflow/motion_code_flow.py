"""Text-to-motion generation by continuous flow over part-specific codebooks."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_blocks import MotionTextDiT, TimestepEmbedder
from .text_encoder import FrozenCLIPTextEncoder, TextCondition
from .vq_tokenizers import build_codeflow_tokenizer


@dataclass
class MotionCodeFlowConfig:
    kv_root: str = "."
    vq_backend: str = "kv_part"
    vq_checkpoint: Optional[str] = None
    vq_partition: Optional[str] = None
    vq_opt_path: Optional[str] = None
    clip_version: str = "ViT-B/32"
    clip_path: Optional[str] = None

    representation: str = "t_concat"
    code_dim: int = 128
    num_parts: int = 6
    num_codes: int = 128
    part_hidden_dim: int = 0
    max_motion_tokens: int = 49
    time_patch: int = 1
    coupling_mode: str = "holder_query"
    holder_depth: int = 2
    holder_mlp_ratio: float = 4.0

    hidden_size: int = 768
    num_heads: int = 12
    depth_double: int = 6
    depth_single: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    cond_drop_prob: float = 0.1
    self_cond_prob: float = 0.5
    use_self_condition: bool = True
    time_schedule: str = "logit_normal"
    denoiser_p_mean: float = -1.5
    denoiser_p_std: float = 0.8
    noise_scale: float = 1.0
    t_eps: float = 1e-4
    latent_norm_mode: str = "none"  # "none", "codebook"
    latent_offset: float = 0.0
    latent_norm_eps: float = 1e-6
    sampling_schedule: str = "uniform"  # "uniform", "logit_normal"
    sampling_method: str = "ode"  # "ode", "sde"
    sde_gamma: float = 0.0
    decode_mode: str = "nearest"  # "nearest", "ids", "continuous"

    terminal_mode: str = "tied_logits"  # "nearest", "tied_logits", "learned_head"
    terminal_tau: float = 1.0
    terminal_tau_mode: str = "fixed"  # "fixed", "codebook_nn"
    terminal_tau_floor: float = 1e-6
    code_ce_t_min: float = 0.0
    code_ce_t_max: float = 1.0
    code_ce_gamma: float = 0.0
    code_ce_normalize: bool = False
    flow_loss_weight: float = 1.0
    terminal_loss_weight: float = 1.0
    clean_loss_weight: float = 0.0


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    schedule: str,
    p_mean: float,
    p_std: float,
) -> torch.Tensor:
    if schedule == "uniform":
        return torch.rand(batch_size, device=device)
    if schedule == "logit_normal":
        return torch.sigmoid(torch.randn(batch_size, device=device) * p_std + p_mean)
    raise ValueError(f"Unknown time schedule: {schedule}")


def _split_rope_dims(hidden_size: int, num_heads: int) -> List[int]:
    head_dim = hidden_size // num_heads
    time_dim = int(round(head_dim * 0.75))
    time_dim = max(2, time_dim - (time_dim % 2))
    part_dim = head_dim - time_dim
    if part_dim < 2:
        part_dim = 2
        time_dim = head_dim - part_dim
    if time_dim % 2:
        time_dim -= 1
        part_dim += 1
    if part_dim % 2:
        part_dim -= 1
        time_dim += 1
    return [time_dim, part_dim]


class MotionCodeFlow(nn.Module):
    """Full DiT flow prior over frozen part-specific VQ code embeddings."""

    def __init__(self, config: MotionCodeFlowConfig) -> None:
        super().__init__()
        if config.representation != "t_concat":
            raise ValueError("holder-query CodeFlow only supports representation='t_concat'")
        if config.coupling_mode != "holder_query":
            raise ValueError("CodeFlow now uses coupling_mode='holder_query' only")
        if config.time_patch != 1:
            raise ValueError("holder-query CodeFlow uses time_patch=1")
        if config.terminal_mode not in {"nearest", "tied_logits", "learned_head"}:
            raise ValueError(f"Unsupported terminal_mode: {config.terminal_mode}")
        if config.terminal_tau_mode not in {"fixed", "codebook_nn"}:
            raise ValueError(f"Unsupported terminal_tau_mode: {config.terminal_tau_mode}")
        if config.latent_norm_mode not in {"none", "codebook"}:
            raise ValueError(f"Unsupported latent_norm_mode: {config.latent_norm_mode}")
        if config.sampling_schedule not in {"uniform", "logit_normal"}:
            raise ValueError(f"Unsupported sampling_schedule: {config.sampling_schedule}")
        if config.sampling_method not in {"ode", "sde"}:
            raise ValueError(f"Unsupported sampling_method: {config.sampling_method}")
        if config.decode_mode not in {"nearest", "ids", "continuous"}:
            raise ValueError(f"Unsupported decode_mode: {config.decode_mode}")
        if config.holder_depth <= 0:
            raise ValueError(f"holder_depth must be positive, got {config.holder_depth}")
        self.config = config

        self.tokenizer = build_codeflow_tokenizer(
            backend=config.vq_backend,
            kv_root=config.kv_root,
            checkpoint_path=config.vq_checkpoint,
            partition_path=config.vq_partition,
            opt_path=config.vq_opt_path,
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

        self.input_code_dim = config.code_dim * (2 if config.use_self_condition else 1)
        conv_in = self.input_code_dim
        output_size = config.time_patch * config.code_dim

        self.input_conv = nn.Sequential(
            nn.Conv1d(conv_in, config.hidden_size, kernel_size=config.time_patch, stride=config.time_patch),
            nn.GELU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1),
        )
        self.part_embed = nn.Parameter(torch.zeros(config.num_parts, config.hidden_size))
        nn.init.normal_(self.part_embed, std=0.02)

        self.timestep_embed = TimestepEmbedder(config.hidden_size)
        self.text_token_proj = nn.Linear(self.text_encoder.width, config.hidden_size)
        self.text_pooled_proj = nn.Sequential(
            nn.Linear(self.text_encoder.output_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.backbone = MotionTextDiT(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            depth_double=config.depth_double,
            depth_single=config.depth_single,
            output_size=output_size,
            num_parts=config.num_parts,
            holder_depth=config.holder_depth,
            holder_mlp_ratio=config.holder_mlp_ratio,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            rope_axes_dims=_split_rope_dims(config.hidden_size, config.num_heads),
        )

        if config.terminal_mode == "learned_head":
            self.learned_heads = nn.ModuleList([
                nn.Linear(config.code_dim, config.num_codes)
                for _ in range(config.num_parts)
            ])
        else:
            self.learned_heads = None

    def _init_latent_stats(self) -> None:
        cfg = self.config
        if cfg.latent_norm_mode == "codebook":
            codebooks = self.tokenizer.codebooks.detach().float()
            mean = codebooks.mean(dim=1)
            std = codebooks.std(dim=1, unbiased=False).clamp_min(float(cfg.latent_norm_eps))
        else:
            mean = torch.zeros(cfg.num_parts, cfg.code_dim, dtype=torch.float32)
            std = torch.ones(cfg.num_parts, cfg.code_dim, dtype=torch.float32)
        self.register_buffer("latent_mean", mean.view(1, 1, cfg.num_parts, cfg.code_dim), persistent=False)
        self.register_buffer("latent_std", std.view(1, 1, cfg.num_parts, cfg.code_dim), persistent=False)

    def _init_terminal_tau(self) -> None:
        cfg = self.config
        if cfg.terminal_tau_mode == "codebook_nn":
            values = []
            for part_idx in range(cfg.num_parts):
                codebook = self.tokenizer.codebooks[part_idx].float()
                dist_sq = torch.cdist(codebook, codebook, p=2.0).square()
                dist_sq.fill_diagonal_(float("inf"))
                nearest = dist_sq.min(dim=1).values
                values.append(torch.median(nearest[torch.isfinite(nearest)]).clamp_min(float(cfg.terminal_tau_floor)))
            tau_parts = torch.stack(values).float()
        else:
            tau_parts = torch.full(
                (cfg.num_parts,),
                max(float(cfg.terminal_tau), float(cfg.terminal_tau_floor)),
                dtype=torch.float32,
            )
        self.register_buffer("terminal_tau_parts", tau_parts, persistent=False)

    @property
    def core_output_weight(self) -> torch.Tensor:
        return self.holder_output.linear.weight

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def holder_output(self):
        return self.backbone.holder_output.linear

    def train(self, mode: bool = True):
        super().train(mode)
        self.tokenizer.vq_model.eval()
        self.text_encoder.clip_model.eval()
        return self

    def trainable_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("tokenizer.") or name.startswith("text_encoder.clip_model."):
                continue
            if param.requires_grad:
                yield param

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        state = self.state_dict()
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("tokenizer.vq_model.")
            and not key.startswith("text_encoder.clip_model.")
        }

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = [
            key for key in missing
            if key.startswith("tokenizer.vq_model.") or key.startswith("text_encoder.clip_model.")
        ]
        bad_missing = [key for key in missing if key not in allowed_missing]
        return bad_missing, list(unexpected)

    def raw_to_model_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Map frozen RVQ codebook embeddings into the DiT training space."""
        mean = self.latent_mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_std.to(device=z.device, dtype=z.dtype)
        return (z - mean) / std + float(self.config.latent_offset)

    def model_to_raw_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Map DiT-space latents back to the frozen KV decoder/codebook space."""
        mean = self.latent_mean.to(device=z.device, dtype=z.dtype)
        std = self.latent_std.to(device=z.device, dtype=z.dtype)
        return (z - float(self.config.latent_offset)) * std + mean

    def _sampling_grid(self, steps: int, device: torch.device) -> torch.Tensor:
        if steps <= 0:
            raise ValueError(f"Sampling steps must be positive, got {steps}")
        cfg = self.config
        if cfg.sampling_schedule == "uniform":
            return torch.linspace(0.0, 1.0, steps + 1, device=device)
        if cfg.sampling_schedule == "logit_normal":
            if steps == 1:
                return torch.tensor([0.0, 1.0], device=device)
            inner = sample_timesteps(
                steps - 1,
                device,
                "logit_normal",
                cfg.denoiser_p_mean,
                cfg.denoiser_p_std,
            ).sort().values
            return torch.cat([inner.new_zeros(1), inner, inner.new_ones(1)], dim=0)
        raise ValueError(f"Unknown sampling schedule: {cfg.sampling_schedule}")

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

    def _pad_time(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        patch = self.config.time_patch
        pad_len = (patch - x.shape[1] % patch) % patch
        if pad_len:
            pad_shape = list(x.shape)
            pad_shape[1] = pad_len
            x = torch.cat([x, x.new_zeros(pad_shape)], dim=1)
        return x, pad_len

    def _patch_mask(self, token_lengths: torch.Tensor, latent_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = lengths_to_mask(token_lengths, latent_len)
        patch = self.config.time_patch
        pad_len = (patch - latent_len % patch) % patch
        if pad_len:
            valid = torch.cat([valid, torch.zeros(valid.shape[0], pad_len, device=valid.device, dtype=torch.bool)], dim=1)
        patched = valid.view(valid.shape[0], -1, patch).any(dim=-1)
        return valid[:, :latent_len], patched

    def _pack_motion(
        self,
        x: torch.Tensor,
        token_lengths: torch.Tensor,
        x_self_cond: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        cfg = self.config
        if cfg.use_self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat([x, x_self_cond], dim=-1)

        latent_len = x.shape[1]
        _, patch_valid = self._patch_mask(token_lengths, latent_len)
        x, pad_len = self._pad_time(x)
        bsz, padded_len, num_parts, dim = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bsz * num_parts, dim, padded_len)
        h = self.input_conv(x)
        patch_len = h.shape[-1]
        h = h.view(bsz, num_parts, cfg.hidden_size, patch_len).permute(0, 3, 1, 2)
        h = h + self.part_embed[None, None]
        tokens = h.reshape(bsz, patch_len * num_parts, cfg.hidden_size)
        motion_valid = patch_valid[:, :, None].expand(bsz, patch_len, num_parts).reshape(bsz, patch_len * num_parts)
        time_ids = torch.arange(patch_len, device=x.device, dtype=torch.float32)
        part_ids = torch.arange(num_parts, device=x.device, dtype=torch.float32)
        tt, pp = torch.meshgrid(time_ids, part_ids, indexing="ij")
        pos = torch.stack([tt, pp], dim=-1).reshape(1, patch_len * num_parts, 2).expand(bsz, -1, -1)
        return tokens, motion_valid, pos, pad_len

    def _unpack_motion(self, y: torch.Tensor, latent_len: int) -> torch.Tensor:
        cfg = self.config
        bsz = y.shape[0]
        patch = cfg.time_patch
        patch_len = y.shape[1] // cfg.num_parts
        y = y.view(bsz, patch_len, cfg.num_parts, patch, cfg.code_dim)
        y = y.permute(0, 1, 3, 2, 4).reshape(bsz, patch_len * patch, cfg.num_parts, cfg.code_dim)
        return y[:, :latent_len]

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
        cfg = self.config
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(z.shape[0])
        timesteps = timesteps.to(device=z.device, dtype=z.dtype)
        token_lengths = token_lengths.to(z.device).long().clamp(min=1, max=z.shape[1])

        text_cond = self._text_condition(texts, drop_prob=text_drop_prob, force_drop=force_text_drop)
        motion_tokens, motion_valid, motion_pos, _ = self._pack_motion(z, token_lengths, x_self_cond)
        cond = self.timestep_embed(timesteps.float()) + text_cond.pooled
        pred = self.backbone(
            motion=motion_tokens,
            text=text_cond.tokens,
            cond=cond,
            motion_valid=motion_valid,
            text_padding_mask=text_cond.padding_mask,
            motion_pos_ids=motion_pos,
            return_hidden=False,
        )
        pred = pred[:, : z.shape[1]]
        valid = lengths_to_mask(token_lengths, z.shape[1]).to(pred.dtype)
        return pred * valid[:, :, None, None]

    def predict_clean_from_velocity(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        while timesteps.ndim < z_t.ndim:
            timesteps = timesteps[..., None]
        return z_t + (1.0 - timesteps).clamp_min(self.config.t_eps) * velocity

    def terminal_logits(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        mode = mode or self.config.terminal_mode
        if mode in {"nearest", "tied_logits"}:
            tau = self.terminal_tau_parts if hasattr(self, "terminal_tau_parts") else self.config.terminal_tau
            return self.tokenizer.codebook_tied_logits(clean_pred, tau=tau)
        if mode == "learned_head":
            if self.learned_heads is None:
                raise RuntimeError("learned_head terminal mode was not initialized")
            logits = []
            for part_idx, head in enumerate(self.learned_heads):
                logits.append(head(clean_pred[:, :, part_idx]))
            return torch.stack(logits, dim=2)
        raise ValueError(f"Unknown terminal mode: {mode}")

    @torch.no_grad()
    def terminal_ids(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        mode = mode or self.config.terminal_mode
        if mode == "nearest":
            return self.tokenizer.nearest_ids(clean_pred)
        return self.terminal_logits(clean_pred, mode=mode).argmax(dim=-1).long()

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

        if x_self_cond is not None:
            x_self_cond = x_self_cond.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
        if x_self_cond is None and cfg.use_self_condition and cfg.self_cond_prob > 0.0 and allow_internal_self_condition:
            with torch.no_grad():
                v_init = self.forward(
                    z_t,
                    t,
                    texts,
                    token_lengths,
                    x_self_cond=None,
                    text_drop_prob=0.0,
                )
                clean_init = self.predict_clean_from_velocity(z_t, t, v_init).detach()
            keep = (torch.rand(bsz, device=target_embeddings.device) < cfg.self_cond_prob).to(target_embeddings.dtype)
            x_self_cond = clean_init * keep[:, None, None, None]

        velocity_pred = self.forward(
            z_t,
            t,
            texts,
            token_lengths,
            x_self_cond=x_self_cond,
            text_drop_prob=cfg.cond_drop_prob,
        )
        velocity_pred_f = velocity_pred.float()
        velocity_target_f = velocity_target.float()
        target_model_f = target_model.float()
        valid_float_f = valid_float.float()
        z_t_f = z_t.float()
        t_f = t.float()

        per_part_flow = (velocity_pred_f - velocity_target_f).square().mean(dim=-1)
        flow_loss = (per_part_flow * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)

        clean_pred = self.predict_clean_from_velocity(z_t_f, t_f, velocity_pred_f)
        clean_loss = (clean_pred - target_model_f).square().mean(dim=-1)
        clean_loss = (clean_loss * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)
        clean_pred_raw = self.model_to_raw_latent(clean_pred)

        terminal_loss = target_model_f.new_zeros(())
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
                code_weight = (
                    (t_f >= float(cfg.code_ce_t_min))
                    & (t_f <= float(cfg.code_ce_t_max))
                ).to(valid_float_f.dtype) * t_f.clamp_min(0.0).pow(float(cfg.code_ce_gamma))
                ce = ce * code_weight[:, None, None]
            terminal_loss = (ce * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)

        with torch.no_grad():
            pred_ids = self.terminal_ids(clean_pred_raw)
            acc = ((pred_ids == target_ids.long()) & valid_parts).sum().float() / valid_parts.sum().float().clamp_min(1.0)
            nn_ids = self.tokenizer.nearest_ids(clean_pred_raw)
            nn_acc = ((nn_ids == target_ids.long()) & valid_parts).sum().float() / valid_parts.sum().float().clamp_min(1.0)

        total = cfg.flow_loss_weight * flow_loss + cfg.terminal_loss_weight * terminal_loss + cfg.clean_loss_weight * clean_loss
        out = {
            "loss": total,
            "flow_loss": flow_loss,
            "terminal_loss": terminal_loss,
            "clean_loss": clean_loss,
            "token_acc": acc,
            "nearest_acc": nn_acc,
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

    @torch.no_grad()
    def sample_embeddings(
        self,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        use_self_condition: bool = True,
    ) -> torch.Tensor:
        cfg = self.config
        text_list = list(texts)
        bsz = len(text_list)
        token_lengths = token_lengths.to(self.device).long()
        latent_len = int(token_lengths.max().item())
        z = torch.randn(
            bsz,
            latent_len,
            cfg.num_parts,
            cfg.code_dim,
            device=self.device,
        ) * cfg.noise_scale
        valid = lengths_to_mask(token_lengths, latent_len).to(z.dtype)
        z = z * valid[:, :, None, None]
        x_self_cond = None
        grid = self._sampling_grid(int(steps), self.device).to(z.dtype)

        def forward_guided(
            z_in: torch.Tensor,
            t_in: torch.Tensor,
            x_sc: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if cond_scale == 1.0:
                v_out = self.forward(
                    z_in,
                    t_in,
                    text_list,
                    token_lengths,
                    x_self_cond=x_sc if use_self_condition else None,
                    text_drop_prob=0.0,
                )
            else:
                z_cat = torch.cat([z_in, z_in], dim=0)
                lengths_cat = torch.cat([token_lengths, token_lengths], dim=0)
                texts_cat = [""] * bsz + text_list
                sc_cat = torch.cat([x_sc, x_sc], dim=0) if x_sc is not None and use_self_condition else None
                v_all = self.forward(
                    z_cat,
                    torch.cat([t_in, t_in], dim=0),
                    texts_cat,
                    lengths_cat,
                    x_self_cond=sc_cat,
                    text_drop_prob=0.0,
                )
                v_uncond, v_cond = v_all.chunk(2, dim=0)
                v_out = v_uncond + float(cond_scale) * (v_cond - v_uncond)
            clean_out = self.predict_clean_from_velocity(z_in, t_in, v_out)
            return v_out, clean_out

        for idx in range(steps):
            t_cur_scalar = grid[idx]
            t_next_scalar = grid[idx + 1]
            dt = t_next_scalar - t_cur_scalar
            z_eval = z
            t_eval_scalar = t_cur_scalar
            if cfg.sampling_method == "sde" and float(cfg.sde_gamma) > 0.0:
                alpha_value = max(0.0, min(1.0, 1.0 - float(cfg.sde_gamma) * float(dt.item())))
                eps = torch.randn_like(z) * cfg.noise_scale
                z_eval = alpha_value * z + (1.0 - alpha_value) * eps
                t_eval_scalar = t_cur_scalar * alpha_value
                dt = t_next_scalar - t_eval_scalar

            t_eval = t_eval_scalar.expand(bsz)
            v, clean = forward_guided(z_eval, t_eval, x_self_cond)
            z = z_eval + dt * v
            z = z * valid[:, :, None, None]
            if cfg.use_self_condition and use_self_condition:
                x_self_cond = clean.detach() * valid[:, :, None, None]

        raw = self.model_to_raw_latent(z)
        return raw * valid[:, :, None, None]

    @torch.no_grad()
    def generate_ids(
        self,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        terminal_mode: Optional[str] = None,
    ) -> torch.Tensor:
        clean = self.sample_embeddings(
            texts,
            token_lengths=token_lengths,
            steps=steps,
            cond_scale=cond_scale,
        )
        return self.terminal_ids(clean, mode=terminal_mode)

    @torch.no_grad()
    def generate_motion(
        self,
        texts: Iterable[str],
        token_lengths: torch.Tensor,
        steps: int = 32,
        cond_scale: float = 3.0,
        terminal_mode: Optional[str] = None,
        decode_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.sample_embeddings(
            texts,
            token_lengths=token_lengths,
            steps=steps,
            cond_scale=cond_scale,
        )
        mode = decode_mode or self.config.decode_mode
        if mode not in {"nearest", "ids", "continuous"}:
            raise ValueError(f"Unsupported decode_mode: {mode}")
        ids = self.terminal_ids(clean, mode=terminal_mode)
        if mode == "continuous":
            motion = self.tokenizer.decode_embeddings(clean)
        else:
            motion = self.tokenizer.decode_ids(ids)
        return motion, ids
