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

from .dit_blocks import BidirectionalTextRefiner, FinalLayer, FrameMotionTextDiT, TimestepEmbedder
from .motion_code_flow import (
    DECODE_MODE,
    TEXT_POOLED_MODES,
    TEXT_REFINER_POOL_MODES,
    MotionCodeFlow,
    MotionCodeFlowConfig,
    lengths_to_mask,
    sample_timesteps,
)
from .text_encoder import TextCondition, build_text_encoder
from .vq_tokenizers import build_codeflow_tokenizer


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
        part_dim = int(config.part_hidden_dim) if int(config.part_hidden_dim) > 0 else int(config.code_dim)
        if config.hidden_size != config.num_parts * part_dim:
            raise ValueError(
                "PartStructuredMotionCodeFlow hidden size must match the grouped latent width: "
                f"hidden_size must be num_parts*part_hidden_dim={config.num_parts * part_dim}, "
                f"got {config.hidden_size}"
            )
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f"hidden_size {config.hidden_size} must be divisible by num_heads={config.num_heads}")
        if config.terminal_mode not in {"none", "nearest", "residual_nearest", "tied_logits", "learned_head"}:
            raise ValueError(f"Unsupported terminal_mode: {config.terminal_mode}")
        if config.latent_norm_mode not in {"none", "codebook", "empirical"}:
            raise ValueError(f"Unsupported latent_norm_mode: {config.latent_norm_mode}")
        if float(config.latent_offset) != 0.0:
            raise ValueError("PartStructuredMotionCodeFlow uses latent_offset=0 to preserve raw codebook metric")
        if config.sampling_schedule not in {"uniform", "logit_normal"}:
            raise ValueError(f"Unsupported sampling_schedule: {config.sampling_schedule}")
        if config.sampling_method not in {"ode", "sde"}:
            raise ValueError(f"Unsupported sampling_method: {config.sampling_method}")
        if config.decode_mode != DECODE_MODE:
            raise ValueError(
                f"decode_mode is fixed to {DECODE_MODE!r} and cannot be configured; "
                f"got {config.decode_mode!r}"
            )
        if config.terminal_tau_mode not in {"fixed", "codebook_nn"}:
            raise ValueError(f"Unsupported terminal_tau_mode: {config.terminal_tau_mode}")
        if config.text_refiner_depth < 0:
            raise ValueError(f"text_refiner_depth must be non-negative, got {config.text_refiner_depth}")
        if config.text_refiner_pool not in TEXT_REFINER_POOL_MODES:
            raise ValueError(
                f"text_refiner_pool must be one of {sorted(TEXT_REFINER_POOL_MODES)}, "
                f"got {config.text_refiner_pool!r}"
            )
        if config.text_pooled_mode not in TEXT_POOLED_MODES:
            raise ValueError(
                f"text_pooled_mode must be one of {sorted(TEXT_POOLED_MODES)}, "
                f"got {config.text_pooled_mode!r}"
            )
        self.config = config

        self.tokenizer = build_codeflow_tokenizer(
            backend=config.vq_backend,
            kv_root=config.kv_root,
            checkpoint_path=config.vq_checkpoint,
            partition_path=config.vq_partition,
            opt_path=config.vq_opt_path,
            num_codes=config.num_codes,
            code_dim=config.code_dim,
            kv_part_target_mode=config.kv_part_target_mode,
            rvq_target_mode=config.rvq_target_mode,
        )
        if self.tokenizer.num_parts != config.num_parts:
            raise ValueError(f"Config num_parts={config.num_parts}, tokenizer has {self.tokenizer.num_parts}")
        if self.tokenizer.num_codes != config.num_codes:
            raise ValueError(f"Config num_codes={config.num_codes}, tokenizer has {self.tokenizer.num_codes}")
        if self.tokenizer.code_dim != config.code_dim:
            raise ValueError(f"Config code_dim={config.code_dim}, tokenizer has {self.tokenizer.code_dim}")
        self._init_latent_stats()
        self._init_terminal_tau()

        self.text_encoder = build_text_encoder(
            config.text_encoder_type,
            clip_version=config.clip_version,
            clip_path=config.clip_path,
            kv_root=config.kv_root,
            text_cache_path=config.text_cache_path,
            qwen_max_length=config.qwen_max_length,
            text_cache_fingerprint=config.text_cache_fingerprint,
            clip_hf_path=config.clip_hf_path,
        )

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
        # Token-only providers expose output_dim=None: no pooled/global text
        # branch is built and AdaLN conditioning reduces to the timestep alone.
        text_pooled_dim = getattr(self.text_encoder, "output_dim", None)
        self.text_pooled_proj = (
            nn.Sequential(
                nn.Linear(text_pooled_dim, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            if text_pooled_dim
            else None
        )
        self.text_token_refiner = (
            BidirectionalTextRefiner(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                depth=config.text_refiner_depth,
                mlp_ratio=config.text_refiner_mlp_ratio,
                dropout=config.dropout,
            )
            if config.text_refiner_depth > 0
            else None
        )
        self._assert_refiner_pool_supported()
        # The two routings build byte-identical parameter sets -- same names,
        # same shapes -- so a strict state-dict load cannot tell them apart and
        # a checkpoint whose stored options were trimmed or rebuilt elsewhere
        # would load a token-trained model in modulation mode with every weight
        # matching and no error.  This buffer travels in the state dict, so the
        # mismatch surfaces as a value check instead of as a worse FID.
        self.register_buffer(
            "text_pooled_mode_code",
            torch.tensor(TEXT_POOLED_MODES.index(config.text_pooled_mode), dtype=torch.long),
            persistent=True,
        )
        # A marker for the prepended sentence token.  Both backbones give every
        # text position the same RoPE id, so the text stream is an unordered bag
        # and position cannot say which token is the sentence vector -- the only
        # other signal is that it came out of a different projection.  Three of
        # the four encoders in this campaign derive their pooled vector from the
        # token stream itself (CLIP-L's is literally one of the tokens), so
        # without a marker the token arm risks degenerating into a no-op.
        # Zero-initialised, so training starts exactly where it would without it.
        # Built only in token mode, which keeps modulation checkpoints loadable.
        self.pooled_type_embed = (
            nn.Parameter(torch.zeros(config.hidden_size))
            if config.text_pooled_mode == "token" and self.text_pooled_proj is not None
            else None
        )
        if config.text_pooled_mode == "token" and self.text_pooled_proj is None:
            raise ValueError(
                "--text_pooled_mode token needs a text provider that returns a pooled "
                f"sentence vector, but {type(self.text_encoder).__name__} "
                f"(text_encoder_type={config.text_encoder_type!r}) reports output_dim=None. "
                "Use a pooled provider, or --text_pooled_mode modulation."
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
            if not hasattr(self.tokenizer, "codebooks"):
                raise ValueError("terminal_tau_mode='codebook_nn' requires a tokenizer with codebooks")
            if self.tokenizer.codebooks.shape[0] != cfg.num_parts:
                raise ValueError(
                    "terminal_tau_mode='codebook_nn' requires one codebook per model part; "
                    f"got codebook parts={self.tokenizer.codebooks.shape[0]} and num_parts={cfg.num_parts}."
                )
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
        timesteps: Optional[torch.Tensor] = None,
        raw_condition: Optional[TextCondition] = None,
    ) -> TextCondition:
        if raw_condition is not None:
            if drop_prob > 0.0 or force_drop:
                raise ValueError("A supplied raw text condition cannot also request text dropout")
            cond = raw_condition
        else:
            cond = self.text_encoder(texts, drop_prob=drop_prob, force_drop=force_drop)
        tokens = self.text_token_proj(cond.tokens)
        if self.text_token_refiner is not None:
            if timesteps is None:
                raise ValueError("timesteps are required when text_refiner_depth > 0")
            tokens = self.text_token_refiner(
                tokens, timesteps, cond.padding_mask,
                self._refiner_content_mask(cond),
            )
        padding_mask = cond.padding_mask
        content_mask = cond.content_mask
        pooled = None
        if self.text_pooled_proj is not None:
            if cond.pooled is None:
                raise ValueError("Text encoder returned no pooled feature but text_pooled_proj exists")
            projected = self.text_pooled_proj(cond.pooled)
            if self.config.text_pooled_mode == "token":
                # The sentence vector becomes an ordinary text token instead of an
                # AdaLN signal, so motion positions attend to it selectively rather
                # than receiving one global affine per block.  AdaLN is left with
                # the timestep alone.
                #
                # Prepended, and prepending is exact rather than a convention we
                # are stuck with: both backbones give every text position the same
                # RoPE id (text_pos = zeros in DoubleStreamBlock.forward and in
                # FrameMotionTextDiT.forward), so where in the text stream it sits
                # cannot change the attention.  Front is simply the only placement
                # that needs no ragged per-row insert.
                #
                # Concatenated *after* the refiner on purpose: the refiner derives
                # its own AdaLN signal from a mean over the tokens it is given, so
                # feeding it this vector would let the sentence vector help produce
                # the modulation that then gates it.
                marked = projected + self.pooled_type_embed
                tokens = torch.cat([marked.unsqueeze(1), tokens], dim=1)
                lead = torch.zeros(
                    padding_mask.shape[0], 1, dtype=torch.bool, device=padding_mask.device
                )
                padding_mask = torch.cat([lead, padding_mask], dim=1)
                if content_mask is not None:
                    # False, not True: content_mask marks *caption* positions, and
                    # its only consumer averages over them to build the refiner's
                    # modulation.  Marking the sentence vector as caption text
                    # would let it help produce the signal that gates it -- the
                    # circularity concat-after-refiner exists to avoid -- the
                    # moment any consumer appears downstream of this concat.
                    content_mask = torch.cat([lead, content_mask], dim=1)
            else:
                pooled = projected
        return TextCondition(
            pooled=pooled,
            tokens=tokens,
            padding_mask=padding_mask,
            content_mask=content_mask,
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
            part_chunk = self.part_inputs[part_idx](part_x)
            part_chunks.append(part_chunk)
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
        raw_text_condition: Optional[TextCondition] = None,
    ) -> torch.Tensor:
        del x_self_cond
        cfg = self.config
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(z.shape[0])
        timesteps = timesteps.to(device=z.device, dtype=z.dtype)
        token_lengths = token_lengths.to(z.device).long().clamp(min=1, max=z.shape[1])

        timestep_cond = self.timestep_embed(timesteps.float())
        text_cond = self._text_condition(
            texts,
            drop_prob=text_drop_prob,
            force_drop=force_text_drop,
            timesteps=timesteps,
            raw_condition=raw_text_condition,
        )
        motion_tokens, motion_valid, motion_pos = self._pack_motion(
            z,
            token_lengths,
        )
        cond = timestep_cond
        if text_cond.pooled is not None:
            cond = cond + text_cond.pooled
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
        if mode == "none":
            raise ValueError("terminal_logits is unavailable when terminal_mode='none'")
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
        target_ids: Optional[torch.Tensor],
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
        has_code_targets = target_ids is not None and bool(getattr(self.tokenizer, "supports_code_metrics", True))
        if getattr(self.tokenizer, "target_mode", "stage") == "sum":
            if target_ids is None:
                raise ValueError("MoMask RVQ sum target requires target_ids")
            expected_quantizers = int(getattr(self.tokenizer, "num_quantizers", target_ids.shape[-1]))
            if num_parts != 1 or target_ids.ndim != 3 or target_ids.shape[-1] != expected_quantizers:
                raise ValueError(
                    "MoMask RVQ sum target expects target_embeddings [B,T,1,D] and "
                    f"target_ids [B,T,{expected_quantizers}], got "
                    f"target_embeddings={tuple(target_embeddings.shape)} target_ids={tuple(target_ids.shape)}"
                )
        token_lengths = token_lengths.to(target_embeddings.device).long().clamp(min=1, max=latent_len)
        valid = lengths_to_mask(token_lengths, latent_len)
        valid_parts = valid[:, :, None].expand(bsz, latent_len, num_parts)
        valid_ids = valid[:, :, None].expand_as(target_ids).bool() if target_ids is not None else None
        valid_float = valid_parts.to(target_embeddings.dtype)
        target_model = self.raw_to_model_latent(target_embeddings)
        cell_weight = valid_float
        loss_denom = cell_weight.float().sum().clamp_min(1.0)

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
        z_t = z_t * valid_float[:, :, :, None]

        x0_pred = self.forward(
            z_t,
            t,
            texts,
            token_lengths,
            x_self_cond=None,
            text_drop_prob=cfg.cond_drop_prob,
        )
        x0_pred_f = x0_pred.float()
        target_model_f = target_model.float()
        valid_float_f = valid_float.float()
        z_t_f = z_t.float()
        t_f = t.float()
        cell_weight_f = cell_weight.float()

        per_part_flow = (x0_pred_f - target_model_f).square().mean(dim=-1)
        flow_loss = (per_part_flow * cell_weight_f).sum() / loss_denom

        # PERMANENT (2026-08-09): the sole training loss is the flow-matching
        # regression on the clean endpoint (x0 MSE) under PREDICTION_TYPE="x0".
        # Terminal/codebook CE and auxiliary losses are deleted and must never
        # be reintroduced.  clean_pred feeds no-grad metrics only.
        clean_pred = x0_pred_f
        clean_pred_raw = self.model_to_raw_latent(clean_pred)

        code_metrics: Dict[str, torch.Tensor] = {}
        if has_code_targets:
            with torch.no_grad():
                pred_ids = self.terminal_ids(clean_pred_raw)
                if pred_ids is None:
                    raise RuntimeError("Code metrics requested but terminal_ids returned None")
                acc = ((pred_ids == target_ids.long()) & valid_ids).sum().float() / valid_ids.sum().float().clamp_min(1.0)
                nn_ids = self.tokenizer.nearest_ids(clean_pred_raw)
                nn_acc = ((nn_ids == target_ids.long()) & valid_ids).sum().float() / valid_ids.sum().float().clamp_min(1.0)
            code_metrics.update({
                "token_acc": acc,
                "nearest_acc": nn_acc,
            })

        total = cfg.flow_loss_weight * flow_loss
        out = {
            "loss": total,
            "flow_loss": flow_loss,
        }
        out.update(code_metrics)
        if include_geometry_metrics and has_code_targets:
            with torch.no_grad():
                pred_ids = self.terminal_ids(clean_pred_raw)
                if pred_ids is None:
                    raise RuntimeError("Geometry metrics requested but terminal_ids returned None")
                code_dist, rank_pct = self.tokenizer.code_id_distances(target_ids.long(), pred_ids.long())
                valid_bool = valid_ids.bool()
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
