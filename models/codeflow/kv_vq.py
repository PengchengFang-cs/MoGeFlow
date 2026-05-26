"""Frozen part-specific VQ tokenizer/decoder wrapper.

This module intentionally treats KV-Control only as a tokenizer/decoder source.
It does not import or expose any control-generation path.
"""

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


VQ_CFG = {
    "dataname": "t2m",
    "batch_size": 256,
    "window_size": 64,
    "total_iter": 300000,
    "warm_up_iter": 1000,
    "lr": 2e-4,
    "lr_scheduler": [200000],
    "gamma": 0.05,
    "weight_decay": 0.0,
    "commit": 0.02,
    "loss_vel": 0.5,
    "recons_loss": "l1_smooth",
    "code_dim": 128,
    "nb_code": 128,
    "mu": 0.99,
    "down_t": 2,
    "stride_t": 2,
    "width": 512,
    "depth": 3,
    "dilation_growth_rate": 3,
    "output_emb_width": 128,
    "vq_act": "relu",
    "vq_norm": None,
    "quantizer": "ema_reset",
    "beta": 1.0,
    "resume_pth": None,
    "resume_gpt": None,
    "out_dir": "output",
    "results_dir": "visual_results/",
    "visual_name": "baseline",
    "exp_name": "exp_debug",
    "print_iter": 200,
    "eval_iter": 5000,
    "seed": 3407,
    "vis_gt": False,
    "nb_vis": 20,
    "sep_uplow": False,
}


def add_kv_control_to_path(kv_root: Path) -> None:
    kv_root = kv_root.expanduser().resolve()
    if str(kv_root) not in sys.path:
        sys.path.insert(0, str(kv_root))


def ids_flat_to_grid(ids_flat: torch.Tensor, num_parts: int) -> torch.Tensor:
    """Convert KV-Control part-major flattened ids to [B, T, P]."""
    if ids_flat.ndim != 2:
        raise ValueError(f"Expected flattened ids [B, P*T], got shape {tuple(ids_flat.shape)}")
    batch, total = ids_flat.shape
    if total % num_parts != 0:
        raise ValueError(f"Cannot reshape id length {total} into {num_parts} parts")
    latent_len = total // num_parts
    return ids_flat.view(batch, num_parts, latent_len).permute(0, 2, 1).contiguous()


def ids_grid_to_flat(ids_grid: torch.Tensor) -> torch.Tensor:
    """Convert [B, T, P] ids to KV-Control part-major flattened ids."""
    if ids_grid.ndim != 3:
        raise ValueError(f"Expected grid ids [B, T, P], got shape {tuple(ids_grid.shape)}")
    return ids_grid.permute(0, 2, 1).contiguous().view(ids_grid.shape[0], -1)


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class PartVQTokenizer(nn.Module):
    """Frozen wrapper around the existing part-specific HumanVQVAE.

    Public shapes:
      - motion: [B, F, 263], normalized with the tokenizer mean/std
      - ids: [B, T, P]
      - embeddings: [B, T, P, D]
    """

    def __init__(
        self,
        kv_root: str,
        checkpoint_path: Optional[str] = None,
        partition_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.kv_root = Path(kv_root).expanduser().resolve()
        add_kv_control_to_path(self.kv_root)

        from kvctrl.models import vqvae

        ckpt_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else (
            self.kv_root / "checkpoints" / "vqvae" / "net_best_fid.pth"
        )
        part_path = Path(partition_path).expanduser().resolve() if partition_path else (
            self.kv_root / "checkpoints" / "vqvae" / "skeleton_partition.json"
        )
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"VQ checkpoint not found: {ckpt_path}")
        if not part_path.is_file():
            raise FileNotFoundError(f"VQ partition not found: {part_path}")

        cfg = dict(VQ_CFG)
        cfg["load_dir_vqvae"] = str(ckpt_path)
        cfg["partition_file"] = str(part_path)
        args = Namespace(**cfg)

        model = vqvae.HumanVQVAE(
            args,
            args.nb_code,
            args.code_dim,
            args.output_emb_width,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm,
        )
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(ckpt, dict) and "net" in ckpt:
            state_dict = ckpt["net"]
        elif isinstance(ckpt, dict) and "vq_model" in ckpt:
            state_dict = ckpt["vq_model"]
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "VQ load mismatch: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.vq_model = model
        self.checkpoint_path = ckpt_path
        self.partition_path = part_path

        with part_path.open("r", encoding="utf-8") as f:
            self.partition = json.load(f)

        codebooks = self._read_codebooks()
        self.num_parts = int(codebooks.shape[0])
        self.num_codes = int(codebooks.shape[1])
        self.code_dim = int(codebooks.shape[2])
        self.register_buffer("codebooks", codebooks, persistent=False)

        if device is not None:
            self.to(device)
        self._verify_static_contract()

    def _verify_static_contract(self) -> None:
        if not hasattr(self.vq_model, "vqvae"):
            raise RuntimeError("KV-Control HumanVQVAE wrapper is missing .vqvae")
        if not hasattr(self.vq_model.vqvae, "forward_decoder"):
            raise RuntimeError("KV-Control VQVAE is missing forward_decoder(ids_grid)")
        if not hasattr(self.vq_model.vqvae, "quantizers"):
            raise RuntimeError("KV-Control VQVAE is missing part quantizers")
        if len(self.vq_model.vqvae.quantizers) != self.num_parts:
            raise RuntimeError(
                f"Expected {self.num_parts} quantizers, got {len(self.vq_model.vqvae.quantizers)}"
            )

    def _read_codebooks(self) -> torch.Tensor:
        books: List[torch.Tensor] = []
        for quantizer in self.vq_model.vqvae.quantizers:
            books.append(quantizer.codebook.detach().float().cpu())
        if not books:
            raise RuntimeError("VQ tokenizer has no part quantizers")
        dims = {tuple(book.shape) for book in books}
        if len(dims) != 1:
            raise RuntimeError(f"Part codebooks have inconsistent shapes: {sorted(dims)}")
        return torch.stack(books, dim=0)

    @property
    def device(self) -> torch.device:
        return self.codebooks.device

    def refresh_codebooks(self) -> None:
        self.codebooks.copy_(self._read_codebooks().to(self.codebooks.device))

    @torch.no_grad()
    def encode_ids(self, motion: torch.Tensor) -> torch.Tensor:
        ids_flat = self.vq_model(motion, type="encode")
        return ids_flat_to_grid(ids_flat.long(), self.num_parts)

    def ids_to_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 3:
            raise ValueError(f"Expected ids [B, T, P], got shape {tuple(ids.shape)}")
        if ids.shape[-1] != self.num_parts:
            raise ValueError(f"Expected {self.num_parts} parts, got {ids.shape[-1]}")
        ids = ids.long()
        parts = []
        for part_idx in range(self.num_parts):
            parts.append(self.codebooks[part_idx].index_select(0, ids[..., part_idx].reshape(-1)).view(
                ids.shape[0], ids.shape[1], self.code_dim
            ))
        return torch.stack(parts, dim=2)

    def codebook_tied_logits(self, z: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Return geometry-tied logits [B, T, P, K]."""
        if z.ndim != 4:
            raise ValueError(f"Expected z [B, T, P, D], got shape {tuple(z.shape)}")
        if z.shape[2] != self.num_parts or z.shape[3] != self.code_dim:
            raise ValueError(
                f"Expected z part/code dims ({self.num_parts}, {self.code_dim}), "
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
        logits = self.codebook_tied_logits(z, tau=1.0)
        return logits.argmax(dim=-1).long()

    @torch.no_grad()
    def decode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 3:
            raise ValueError(f"Expected ids [B, T, P], got shape {tuple(ids.shape)}")
        if ids.shape[-1] != self.num_parts:
            raise ValueError(f"Expected {self.num_parts} parts, got {ids.shape[-1]}")
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.num_codes):
            raise ValueError(f"Code ids must be in [0, {self.num_codes}), got min={ids.min()} max={ids.max()}")
        # KV-Control's current VQVAE_251.forward_decoder contract is grid ids [B, T, P].
        # Its encode path returns flat part-major ids, which encode_ids converts to this grid layout.
        return self.vq_model.vqvae.forward_decoder(ids.long())

    def decode_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        """Decode continuous part-code embeddings [B, T, P, D] through the frozen KV decoder.

        Gradients are allowed to flow to ``z``. The underlying KV decoder remains
        frozen because its parameters have ``requires_grad=False``.
        """
        if z.ndim != 4:
            raise ValueError(f"Expected embeddings [B, T, P, D], got shape {tuple(z.shape)}")
        if z.shape[2] != self.num_parts or z.shape[3] != self.code_dim:
            raise ValueError(
                f"Expected embedding part/code dims ({self.num_parts}, {self.code_dim}), "
                f"got ({z.shape[2]}, {z.shape[3]})"
            )
        parts = [z[:, :, part_idx].permute(0, 2, 1).contiguous() for part_idx in range(self.num_parts)]
        x_quantized = torch.cat(parts, dim=1)
        x_decoder = self.vq_model.vqvae.decoder(x_quantized)
        return self.vq_model.vqvae.postprocess(x_decoder)

    @torch.no_grad()
    def encode(self, motion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.encode_ids(motion)
        return ids, self.ids_to_embeddings(ids)

    @torch.no_grad()
    def code_id_distances(self, target_ids: torch.Tensor, pred_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return codebook distance and distance-rank percentile for id pairs.

        Both inputs are [B, T, P]. The rank percentile is 0 for identical codes and
        approaches 1 for the farthest code in the target code's part-specific codebook.
        """
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
        """Run structural and optional real-motion checks for the frozen VQ boundary."""
        self._verify_static_contract()
        summary: Dict[str, object] = {
            "checkpoint_path": str(self.checkpoint_path),
            "partition_path": str(self.partition_path),
            "num_parts": self.num_parts,
            "num_codes": self.num_codes,
            "code_dim": self.code_dim,
            "decoder_id_layout": "grid_B_T_P",
            "encode_flat_layout": "part_major_B_PxT",
            "codebook_shape": list(self.codebooks.shape),
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

        ids_flat = self.vq_model(motion, type="encode").long()
        ids_grid = ids_flat_to_grid(ids_flat, self.num_parts)
        ids_via_public = self.encode_ids(motion)
        if not torch.equal(ids_grid, ids_via_public):
            raise RuntimeError("encode_ids is not consistent with KV-Control flat part-major ids")

        embeddings = self.ids_to_embeddings(ids_grid)
        nearest = self.nearest_ids(embeddings)
        nearest_match = (nearest == ids_grid).float().mean()
        if not torch.equal(nearest, ids_grid):
            mismatch = int((nearest != ids_grid).sum().item())
            raise RuntimeError(f"ids_to_embeddings/nearest_ids contract failed: {mismatch} mismatched ids")

        decoded = self.decode_ids(ids_grid)
        if decoded.ndim != 3 or decoded.shape[0] != motion.shape[0] or decoded.shape[-1] != motion.shape[-1]:
            raise RuntimeError(
                f"Decoded motion shape {tuple(decoded.shape)} is incompatible with input {tuple(motion.shape)}"
            )
        common_len = min(decoded.shape[1], motion.shape[1])
        recon_abs = (decoded[:, :common_len] - motion[:, :common_len]).abs()
        recon_sq = (decoded[:, :common_len] - motion[:, :common_len]).square()
        if lengths is not None:
            frame_mask = _lengths_to_mask(lengths.clamp(min=1, max=common_len), common_len)
            weight = frame_mask[:, :, None].to(recon_abs.dtype)
            denom = weight.sum().mul(motion.shape[-1]).clamp_min(1.0)
            recon_l1 = (recon_abs * weight).sum() / denom
            recon_mse = (recon_sq * weight).sum() / denom
        else:
            recon_l1 = recon_abs.mean()
            recon_mse = recon_sq.mean()
        if not torch.isfinite(recon_l1) or not torch.isfinite(recon_mse):
            raise RuntimeError("VQ round-trip reconstruction produced non-finite error")

        summary.update({
            "input_motion_shape": list(motion.shape),
            "encoded_flat_shape": list(ids_flat.shape),
            "encoded_grid_shape": list(ids_grid.shape),
            "embedding_shape": list(embeddings.shape),
            "decoded_motion_shape": list(decoded.shape),
            "nearest_embedding_id_match": float(nearest_match.detach().cpu()),
            "roundtrip_l1": float(recon_l1.detach().cpu()),
            "roundtrip_mse": float(recon_mse.detach().cpu()),
        })
        return summary


def load_part_vq_tokenizer(
    kv_root: str,
    checkpoint_path: Optional[str] = None,
    partition_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> PartVQTokenizer:
    return PartVQTokenizer(
        kv_root=kv_root,
        checkpoint_path=checkpoint_path,
        partition_path=partition_path,
        device=device,
    )
