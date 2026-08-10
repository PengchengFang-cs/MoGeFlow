#!/usr/bin/env python3
"""E1: off-lattice diagnostics for PS-CF terminal latent states.

Samples the HumanML3D test set with a trained PartStructuredMotionCodeFlow
checkpoint (kv_part backend), takes the TERMINAL latent state of the flow
BEFORE decoding, and measures how far each part-group latent sits from its
nearest codebook entry, relative to the codebook's own nearest-neighbor
spacing.

The terminal state is obtained through ``model.sample_embeddings``: that is
exactly the integration loop ``generate_motion`` runs, already mapped back to
raw codebook space via ``model_to_raw_latent`` and masked to valid frames --
the only steps it omits are terminal-id projection and decoding, which is
precisely the point.  No model files are modified.

Per part-group p and valid frame t:
    d_tp   = min_k || y_tp - e_pk ||_2
    s_p    = median_k min_{k' != k} || e_pk - e_pk' ||_2
    delta  = d_tp / (0.5 * s_p)

Writes JSON to eval_results/diag_20260809/offlattice_delta.json and prints a
readable per-group summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(TOOLS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.codeflow import PartStructuredMotionCodeFlow
from models.codeflow.eval_t2m_cli import build_eval_loader, load_codeflow_model, make_device
from motion_code_geometry_diagnostics_formal import FORMAL_PART_NAMES
from utils.fixseed import fixseed


DEFAULT_CHECKPOINT = str(
    REPO_ROOT
    / "checkpoints/t2m/hml3d_20260807_nopool_gate/model/gate/gate_ep0300_fid0.0750_top30.8707.pt"
)
DEFAULT_OUT = str(REPO_ROOT / "eval_results/diag_20260809/offlattice_delta.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Off-lattice distance diagnostics for PS-CF terminal latents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_prompts", type=int, default=512, help="Prompts to sample; 0 uses the full split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=96, help="Euler steps.")
    parser.add_argument("--cond_scale", type=float, default=6.0)
    parser.add_argument("--no_ema", action="store_true", help="Use raw model weights instead of EMA.")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=-1)

    # Artifact overrides consumed by the eval_t2m_cli loaders; empty strings
    # keep the checkpoint-recorded paths.
    parser.add_argument("--dataset_opt_path", type=str, default="./checkpoints/t2m/Comp_v6_KLD005/opt.txt")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--glove_dir", type=str, default="./glove")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--kv_root", type=str, default="")
    parser.add_argument("--vq_checkpoint", type=str, default="")
    parser.add_argument("--vq_partition", type=str, default="")
    parser.add_argument("--mean_path", type=str, default="")
    parser.add_argument("--std_path", type=str, default="")
    parser.add_argument("--clip_path", type=str, default="")
    parser.add_argument("--text_cache_path", type=str, default="")
    parser.add_argument("--unit_length", type=int, default=0)
    return parser.parse_args()


def codebook_nn_spacing(codebooks: torch.Tensor) -> torch.Tensor:
    """Per part: median over codes of the nearest-other-code L2 distance."""
    spacings = []
    for part_idx in range(codebooks.shape[0]):
        book = codebooks[part_idx].float()
        dist = torch.cdist(book, book, p=2.0)
        dist.fill_diagonal_(float("inf"))
        nearest = dist.min(dim=1).values
        spacings.append(torch.median(nearest[torch.isfinite(nearest)]))
    return torch.stack(spacings)


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
    }


def delta_summary(d_values: np.ndarray, spacing: float) -> Dict[str, object]:
    half_spacing = 0.5 * spacing
    delta = d_values / max(half_spacing, 1e-12)
    return {
        "count": int(d_values.size),
        "d": summarize(d_values),
        "delta": summarize(delta),
        "frac_delta_lt_0.5": float(np.mean(delta < 0.5)),
        "frac_delta_lt_1": float(np.mean(delta < 1.0)),
        "frac_delta_ge_1": float(np.mean(delta >= 1.0)),
    }


def part_group_names(tokenizer, num_parts: int) -> List[str]:
    partition = getattr(tokenizer, "partition", None) or {}
    names = partition.get("part_names")
    if isinstance(names, list) and len(names) == num_parts:
        return [str(name) for name in names]
    if num_parts == len(FORMAL_PART_NAMES):
        return list(FORMAL_PART_NAMES)
    return [f"part_{idx}" for idx in range(num_parts)]


def main() -> None:
    args = parse_args()
    device = make_device(args)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    model, opt, _ckpt, weight_source = load_codeflow_model(
        checkpoint_path, args, device, model_cls=PartStructuredMotionCodeFlow
    )
    if str(getattr(opt, "vq_backend", "kv_part")) != "kv_part":
        raise ValueError(
            f"offlattice_delta expects a kv_part checkpoint, got vq_backend={opt.vq_backend!r}"
        )
    if not hasattr(model.tokenizer, "codebooks"):
        raise RuntimeError("Tokenizer exposes no codebooks; cannot measure lattice distances")

    loader, dataset, _eval_opt = build_eval_loader(args, opt, device)
    unit_length = int(getattr(opt, "unit_length", 4))
    num_parts = int(model.config.num_parts)
    names = part_group_names(model.tokenizer, num_parts)

    codebooks = model.tokenizer.codebooks.detach().float().to(device)
    spacing = codebook_nn_spacing(codebooks).cpu().numpy()

    fixseed(args.seed)
    model.eval()
    per_part_d: List[List[np.ndarray]] = [[] for _ in range(num_parts)]
    num_prompts = 0
    with torch.no_grad():
        for batch in loader:
            _w, _p, captions, _sl, _pose, m_length, _tok = batch
            m_length = m_length.to(device).long()
            token_lengths = (m_length // unit_length).clamp(min=1)
            # Terminal latent before decoding, already in raw codebook space.
            terminal = model.sample_embeddings(
                list(captions),
                token_lengths=token_lengths,
                steps=int(args.steps),
                cond_scale=float(args.cond_scale),
            )
            latent_len = terminal.shape[1]
            valid = (
                torch.arange(latent_len, device=device)[None, :]
                < token_lengths.clamp(max=latent_len)[:, None]
            )
            for part_idx in range(num_parts):
                y = terminal[:, :, part_idx, :].float()
                flat = y[valid]
                if flat.numel() == 0:
                    continue
                dist = torch.cdist(flat, codebooks[part_idx], p=2.0)
                d_min = dist.min(dim=1).values
                per_part_d[part_idx].append(d_min.cpu().numpy().astype(np.float32))
            num_prompts += len(captions)
            print(f"[sample] {num_prompts} prompts processed", flush=True)
            if args.max_prompts > 0 and num_prompts >= args.max_prompts:
                break

    if num_prompts == 0:
        raise RuntimeError("Eval loader produced zero samples")

    groups: Dict[str, object] = {}
    all_d: List[np.ndarray] = []
    all_delta: List[np.ndarray] = []
    for part_idx in range(num_parts):
        d_values = np.concatenate(per_part_d[part_idx]) if per_part_d[part_idx] else np.zeros(0, np.float32)
        if d_values.size == 0:
            raise RuntimeError(f"No valid frames collected for part {part_idx}")
        s_p = float(spacing[part_idx])
        groups[names[part_idx]] = {
            "part": part_idx,
            "codebook_nn_spacing": s_p,
            **delta_summary(d_values, s_p),
        }
        all_d.append(d_values)
        all_delta.append(d_values / max(0.5 * s_p, 1e-12))

    d_pooled = np.concatenate(all_d)
    delta_pooled = np.concatenate(all_delta)
    overall = {
        "count": int(d_pooled.size),
        "d": summarize(d_pooled),
        "delta": summarize(delta_pooled),
        "frac_delta_lt_0.5": float(np.mean(delta_pooled < 0.5)),
        "frac_delta_lt_1": float(np.mean(delta_pooled < 1.0)),
        "frac_delta_ge_1": float(np.mean(delta_pooled >= 1.0)),
    }

    payload = {
        "checkpoint": str(checkpoint_path),
        "weight_source": weight_source,
        "split": "test",
        "dataset_size": len(dataset),
        "num_prompts": int(num_prompts),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "cond_scale": float(args.cond_scale),
        "unit_length": unit_length,
        "vq_backend": str(getattr(opt, "vq_backend", "kv_part")),
        "vq_checkpoint": str(getattr(opt, "vq_checkpoint", "")),
        "latent_norm_mode": str(getattr(opt, "latent_norm_mode", "")),
        "num_parts": num_parts,
        "num_codes": int(model.config.num_codes),
        "code_dim": int(model.config.code_dim),
        "delta_definition": "delta = d_tp / (0.5 * s_p); s_p = median_k min_{k'!=k} ||e_pk - e_pk'||_2",
        "groups": groups,
        "overall": overall,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = f"{'group':<20} {'s_p':>8} {'d_mean':>8} {'d_med':>8} {'d_p90':>8} {'dl_mean':>8} {'dl_med':>8} {'dl_p90':>8} {'<0.5':>6} {'<1':>6} {'>=1':>6}"
    print()
    print(f"Off-lattice delta: {num_prompts} prompts, steps={args.steps}, cfg={args.cond_scale}, {weight_source} weights")
    print(header)
    print("-" * len(header))
    for name in names:
        g = groups[name]
        print(
            f"{name:<20} {g['codebook_nn_spacing']:>8.4f} "
            f"{g['d']['mean']:>8.4f} {g['d']['median']:>8.4f} {g['d']['p90']:>8.4f} "
            f"{g['delta']['mean']:>8.4f} {g['delta']['median']:>8.4f} {g['delta']['p90']:>8.4f} "
            f"{g['frac_delta_lt_0.5']:>6.3f} {g['frac_delta_lt_1']:>6.3f} {g['frac_delta_ge_1']:>6.3f}"
        )
    print("-" * len(header))
    print(
        f"{'overall':<20} {'-':>8} "
        f"{overall['d']['mean']:>8.4f} {overall['d']['median']:>8.4f} {overall['d']['p90']:>8.4f} "
        f"{overall['delta']['mean']:>8.4f} {overall['delta']['median']:>8.4f} {overall['delta']['p90']:>8.4f} "
        f"{overall['frac_delta_lt_0.5']:>6.3f} {overall['frac_delta_lt_1']:>6.3f} {overall['frac_delta_ge_1']:>6.3f}"
    )
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
