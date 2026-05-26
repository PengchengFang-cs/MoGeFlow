"""Continuous-decode CodeFlow variant.

This variant keeps the holder-query flow backbone, but trains and samples as a
continuous latent generator rather than a terminal code classifier.
"""

from typing import Dict, Iterable, Optional, Tuple

import torch

from .motion_code_flow import MotionCodeFlow, lengths_to_mask, sample_timesteps


class ContinuousMotionCodeFlow(MotionCodeFlow):
    """Flow over RVQ code embeddings with continuous KV decoder output."""

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
        total = cfg.flow_loss_weight * flow_loss + cfg.clean_loss_weight * clean_loss

        with torch.no_grad():
            pred_ids = self.tokenizer.nearest_ids(clean_pred_raw)
            acc = ((pred_ids == target_ids.long()) & valid_parts).sum().float() / valid_parts.sum().float().clamp_min(1.0)

        out = {
            "loss": total,
            "flow_loss": flow_loss,
            "terminal_loss": terminal_loss,
            "clean_loss": clean_loss,
            "token_acc": acc,
            "nearest_acc": acc,
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
