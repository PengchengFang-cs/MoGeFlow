"""Frozen MoMask RVQVAE tokenizer/decoder wrapper."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.vq.model import RVQVAE


def _parse_value(value: str):
    value = value.strip()
    if value in {"True", "False"}:
        return value == "True"
    if value == "None":
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_momask_opt(path: Path) -> Namespace:
    values: Dict[str, object] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key.startswith("-") or not key:
                continue
            values[key] = _parse_value(value)
    required = ["nb_code", "code_dim", "num_quantizers", "down_t", "stride_t", "width", "depth"]
    missing = [key for key in required if key not in values]
    if missing:
        raise RuntimeError(f"MoMask VQ opt file is missing keys: {missing}")
    values.setdefault("shared_codebook", False)
    values.setdefault("quantize_dropout_prob", 0.0)
    values.setdefault("dilation_growth_rate", 3)
    values.setdefault("vq_act", "relu")
    values.setdefault("vq_norm", None)
    values.setdefault("mu", 0.99)
    return Namespace(**values)


class MoMaskRVQTokenizer(nn.Module):
    """Frozen wrapper around the original MoMask residual VQ-VAE.

    Public shapes match the CodeFlow tokenizer boundary:
      - motion: [B, F, 263], normalized with the RVQVAE mean/std
      - ids: [B, T, Q]
      - embeddings: [B, T, Q, D]

    Here Q is the number of residual quantizer layers, not a body-part axis.
    """

    def __init__(
        self,
        checkpoint_path: str,
        opt_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"MoMask VQ checkpoint not found: {ckpt_path}")
        if opt_path:
            opt_file = Path(opt_path).expanduser().resolve()
        else:
            opt_file = ckpt_path.parents[1] / "opt.txt"
        if not opt_file.is_file():
            raise FileNotFoundError(f"MoMask VQ opt file not found: {opt_file}")

        args = load_momask_opt(opt_file)
        input_width = 263
        model = RVQVAE(
            args,
            input_width=input_width,
            nb_code=int(args.nb_code),
            code_dim=int(args.code_dim),
            output_emb_width=int(args.code_dim),
            down_t=int(args.down_t),
            stride_t=int(args.stride_t),
            width=int(args.width),
            depth=int(args.depth),
            dilation_growth_rate=int(args.dilation_growth_rate),
            activation=str(args.vq_act),
            norm=args.vq_norm,
        )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "net" in ckpt:
            state_dict = ckpt["net"]
        elif isinstance(ckpt, dict) and "vq_model" in ckpt:
            state_dict = ckpt["vq_model"]
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in unexpected if not key.endswith(("code_sum", "code_count"))]
        if unexpected:
            raise RuntimeError(f"MoMask VQ load mismatch: unexpected={unexpected[:8]}")
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self.vq_model = model
        self.checkpoint_path = ckpt_path
        self.opt_path = opt_file
        self.partition_path = None
        self.num_parts = int(args.num_quantizers)
        self.num_codes = int(args.nb_code)
        self.code_dim = int(args.code_dim)
        self.register_buffer("codebooks", self._read_codebooks(), persistent=False)
        if missing:
            self.missing_state_keys = tuple(missing)
        else:
            self.missing_state_keys = tuple()

        if device is not None:
            self.to(device)
        self._verify_static_contract()

    @property
    def device(self) -> torch.device:
        return self.codebooks.device

    def _read_codebooks(self) -> torch.Tensor:
        books: List[torch.Tensor] = []
        for layer in self.vq_model.quantizer.layers:
            books.append(layer.codebook.detach().float().cpu())
        if not books:
            raise RuntimeError("MoMask RVQVAE has no residual quantizer layers")
        dims = {tuple(book.shape) for book in books}
        if len(dims) != 1:
            raise RuntimeError(f"MoMask RVQ codebooks have inconsistent shapes: {sorted(dims)}")
        return torch.stack(books, dim=0)

    def refresh_codebooks(self) -> None:
        self.codebooks.copy_(self._read_codebooks().to(self.codebooks.device))

    def _verify_static_contract(self) -> None:
        if not hasattr(self.vq_model, "encode"):
            raise RuntimeError("MoMask RVQVAE is missing encode(x)")
        if not hasattr(self.vq_model, "forward_decoder"):
            raise RuntimeError("MoMask RVQVAE is missing forward_decoder(ids)")
        if not hasattr(self.vq_model, "decoder"):
            raise RuntimeError("MoMask RVQVAE is missing decoder")
        if self.codebooks.shape != (self.num_parts, self.num_codes, self.code_dim):
            raise RuntimeError(
                "MoMask RVQ codebook shape mismatch: "
                f"{tuple(self.codebooks.shape)} vs ({self.num_parts}, {self.num_codes}, {self.code_dim})"
            )

    @torch.no_grad()
    def encode_ids(self, motion: torch.Tensor) -> torch.Tensor:
        ids, _all_codes = self.vq_model.encode(motion)
        if ids.ndim != 3:
            raise RuntimeError(f"Expected MoMask ids [B,T,Q], got {tuple(ids.shape)}")
        return ids.long()

    def ids_to_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 3:
            raise ValueError(f"Expected ids [B, T, Q], got shape {tuple(ids.shape)}")
        if ids.shape[-1] != self.num_parts:
            raise ValueError(f"Expected {self.num_parts} residual layers, got {ids.shape[-1]}")
        ids = ids.long()
        parts = []
        for part_idx in range(self.num_parts):
            part = self.codebooks[part_idx].index_select(0, ids[..., part_idx].reshape(-1))
            parts.append(part.view(ids.shape[0], ids.shape[1], self.code_dim))
        return torch.stack(parts, dim=2)

    def codebook_tied_logits(self, z: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"Expected z [B, T, Q, D], got shape {tuple(z.shape)}")
        if z.shape[2] != self.num_parts or z.shape[3] != self.code_dim:
            raise ValueError(
                f"Expected z residual/code dims ({self.num_parts}, {self.code_dim}), "
                f"got ({z.shape[2]}, {z.shape[3]})"
            )
        tau_tensor = torch.as_tensor(tau, device=z.device, dtype=torch.float32).flatten()
        if tau_tensor.numel() == 1:
            tau_tensor = tau_tensor.expand(self.num_parts)
        elif tau_tensor.numel() != self.num_parts:
            raise ValueError(f"tau must be scalar or length {self.num_parts}, got shape {tuple(tau_tensor.shape)}")
        tau_tensor = tau_tensor.clamp_min(1e-8)
        logits = []
        for part_idx in range(self.num_parts):
            z_part = z[:, :, part_idx].float()
            codebook = self.codebooks[part_idx].float()
            dist = (
                z_part.square().sum(dim=-1, keepdim=True)
                - 2.0 * torch.matmul(z_part, codebook.t())
                + codebook.square().sum(dim=-1)[None, None]
            ).clamp_min_(0.0)
            logits.append(-dist / tau_tensor[part_idx])
        return torch.stack(logits, dim=2)

    @torch.no_grad()
    def nearest_ids(self, z: torch.Tensor) -> torch.Tensor:
        return self.codebook_tied_logits(z, tau=1.0).argmax(dim=-1).long()

    @torch.no_grad()
    def decode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 3:
            raise ValueError(f"Expected ids [B, T, Q], got shape {tuple(ids.shape)}")
        if ids.shape[-1] != self.num_parts:
            raise ValueError(f"Expected {self.num_parts} residual layers, got {ids.shape[-1]}")
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.num_codes):
            raise ValueError(f"Code ids must be in [0, {self.num_codes}), got min={ids.min()} max={ids.max()}")
        return self.vq_model.forward_decoder(ids.long())

    def decode_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"Expected embeddings [B, T, Q, D], got shape {tuple(z.shape)}")
        if z.shape[2] != self.num_parts or z.shape[3] != self.code_dim:
            raise ValueError(
                f"Expected embedding residual/code dims ({self.num_parts}, {self.code_dim}), "
                f"got ({z.shape[2]}, {z.shape[3]})"
            )
        latent = z.sum(dim=2).permute(0, 2, 1).contiguous()
        return self.vq_model.decoder(latent)

    @torch.no_grad()
    def encode(self, motion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.encode_ids(motion)
        return ids, self.ids_to_embeddings(ids)

    @torch.no_grad()
    def code_id_distances(self, target_ids: torch.Tensor, pred_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if target_ids.shape != pred_ids.shape:
            raise ValueError(f"target/pred id shape mismatch: {tuple(target_ids.shape)} vs {tuple(pred_ids.shape)}")
        if target_ids.ndim != 3 or target_ids.shape[-1] != self.num_parts:
            raise ValueError(f"Expected ids [B, T, {self.num_parts}], got {tuple(target_ids.shape)}")
        target_ids = target_ids.long()
        pred_ids = pred_ids.long()
        dists: List[torch.Tensor] = []
        rank_pcts: List[torch.Tensor] = []
        denom = max(self.num_codes - 1, 1)
        for part_idx in range(self.num_parts):
            dist_mat = torch.cdist(
                self.codebooks[part_idx].float(),
                self.codebooks[part_idx].float(),
                p=2.0,
            )
            target = target_ids[..., part_idx]
            pred = pred_ids[..., part_idx]
            pair_dist = dist_mat[target, pred]
            target_rows = dist_mat[target]
            rank = (target_rows <= pair_dist[..., None] + 1e-8).sum(dim=-1) - 1
            dists.append(pair_dist)
            rank_pcts.append((rank.float() / float(denom)).clamp_(0.0, 1.0))
        return torch.stack(dists, dim=2), torch.stack(rank_pcts, dim=2)

    @torch.no_grad()
    def verify_contract(
        self,
        motion: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        max_samples: int = 2,
    ) -> Dict[str, object]:
        self._verify_static_contract()
        summary: Dict[str, object] = {
            "checkpoint_path": str(self.checkpoint_path),
            "opt_path": str(self.opt_path),
            "backend": "momask_rvq",
            "num_parts": self.num_parts,
            "num_codes": self.num_codes,
            "code_dim": self.code_dim,
            "decoder_id_layout": "rvq_grid_B_T_Q",
            "codebook_shape": list(self.codebooks.shape),
            "missing_state_keys": list(self.missing_state_keys[:8]),
        }
        if motion is None:
            return summary

        if motion.ndim != 3:
            raise ValueError(f"Expected motion [B, F, 263], got shape {tuple(motion.shape)}")
        if motion.shape[-1] != 263:
            raise ValueError(f"Expected HumanML3D 263-dim motion features, got {motion.shape[-1]}")
        motion = motion[:max_samples].to(self.device)
        if lengths is not None:
            lengths = lengths[: motion.shape[0]].to(self.device).long()

        ids = self.encode_ids(motion)
        embeddings = self.ids_to_embeddings(ids)
        nearest = self.nearest_ids(embeddings)
        nearest_match = (nearest == ids).float().mean()
        if not torch.equal(nearest, ids):
            mismatch = int((nearest != ids).sum().item())
            raise RuntimeError(f"ids_to_embeddings/nearest_ids contract failed: {mismatch} mismatched ids")

        decoded = self.decode_ids(ids)
        if decoded.ndim != 3 or decoded.shape[0] != motion.shape[0] or decoded.shape[-1] != motion.shape[-1]:
            raise RuntimeError(
                f"Decoded motion shape {tuple(decoded.shape)} is incompatible with input {tuple(motion.shape)}"
            )
        common_len = min(decoded.shape[1], motion.shape[1])
        recon_abs = (decoded[:, :common_len] - motion[:, :common_len]).abs()
        recon_sq = (decoded[:, :common_len] - motion[:, :common_len]).square()
        if lengths is not None:
            frame_mask = torch.arange(common_len, device=motion.device)[None, :] < lengths.clamp(
                min=1, max=common_len
            )[:, None]
            weight = frame_mask[:, :, None].to(recon_abs.dtype)
            denom = weight.sum().mul(motion.shape[-1]).clamp_min(1.0)
            recon_l1 = (recon_abs * weight).sum() / denom
            recon_mse = (recon_sq * weight).sum() / denom
        else:
            recon_l1 = recon_abs.mean()
            recon_mse = recon_sq.mean()
        if not torch.isfinite(recon_l1) or not torch.isfinite(recon_mse):
            raise RuntimeError("MoMask RVQ round-trip reconstruction produced non-finite error")

        summary.update({
            "input_motion_shape": list(motion.shape),
            "encoded_grid_shape": list(ids.shape),
            "embedding_shape": list(embeddings.shape),
            "decoded_motion_shape": list(decoded.shape),
            "nearest_embedding_id_match": float(nearest_match.detach().cpu()),
            "roundtrip_l1": float(recon_l1.detach().cpu()),
            "roundtrip_mse": float(recon_mse.detach().cpu()),
        })
        return summary


def load_momask_rvq_tokenizer(
    checkpoint_path: str,
    opt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> MoMaskRVQTokenizer:
    return MoMaskRVQTokenizer(
        checkpoint_path=checkpoint_path,
        opt_path=opt_path,
        device=device,
    )
