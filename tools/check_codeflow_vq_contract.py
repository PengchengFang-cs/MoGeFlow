#!/usr/bin/env python3
"""Preflight the frozen KV-Control VQ boundary used by CodeFlow."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.codeflow.kv_vq import PartVQTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-root", type=str, default=".")
    parser.add_argument("--data-root", type=str, default="dataset/HumanML3D")
    parser.add_argument("--vq-checkpoint", type=str, default="")
    parser.add_argument("--vq-partition", type=str, default="")
    parser.add_argument("--mean-path", type=str, default="")
    parser.add_argument("--std-path", type=str, default="")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--motion-length", type=int, default=196)
    parser.add_argument("--max-motions", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-non-vq-stats", action="store_true")
    return parser.parse_args()


def load_stats(args, kv_root: Path):
    mean_path = Path(args.mean_path).expanduser().resolve() if args.mean_path else (
        kv_root / "checkpoints" / "stats" / "mean.npy"
    )
    std_path = Path(args.std_path).expanduser().resolve() if args.std_path else (
        kv_root / "checkpoints" / "stats" / "std.npy"
    )
    mean = np.load(mean_path)
    std = np.load(std_path)
    if mean.shape != (263,) or std.shape != (263,):
        raise RuntimeError(f"Expected HumanML3D stats shape (263,), got mean={mean.shape} std={std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or float(np.min(std)) <= 0.0:
        raise RuntimeError("Invalid normalization stats")

    ref_mean_path = kv_root / "checkpoints" / "stats" / "mean.npy"
    ref_std_path = kv_root / "checkpoints" / "stats" / "std.npy"
    matched = False
    if ref_mean_path.is_file() and ref_std_path.is_file():
        matched = bool(
            np.allclose(mean, np.load(ref_mean_path), rtol=0.0, atol=1e-6)
            and np.allclose(std, np.load(ref_std_path), rtol=0.0, atol=1e-6)
        )
    if not args.allow_non_vq_stats and not matched:
        raise RuntimeError("The supplied mean/std do not match KV-Control checkpoints/stats")
    return mean.astype(np.float32), std.astype(np.float32), {
        "mean_path": str(mean_path),
        "std_path": str(std_path),
        "matched_released_vq_stats": matched,
        "std_min": float(np.min(std)),
        "std_max": float(np.max(std)),
    }


def load_motions(args, mean: np.ndarray, std: np.ndarray):
    data_root = Path(args.data_root).expanduser().resolve()
    split_path = data_root / f"{args.split}.txt"
    motion_dir = data_root / "new_joint_vecs"
    motions = []
    lengths = []
    names = []
    with split_path.open("r", encoding="utf-8") as f:
        split_ids = [line.strip() for line in f if line.strip()]
    for name in split_ids:
        if len(motions) >= args.max_motions:
            break
        path = motion_dir / f"{name}.npy"
        if not path.is_file():
            continue
        motion = np.load(path)
        if motion.ndim != 2 or motion.shape[1] != 263 or len(motion) < 4:
            continue
        valid_len = min(len(motion), args.motion_length)
        motion = motion[: args.motion_length].astype(np.float32, copy=False)
        if len(motion) < args.motion_length:
            pad = np.zeros((args.motion_length - len(motion), 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        motions.append((motion - mean) / std)
        lengths.append(valid_len)
        names.append(name)
    if not motions:
        raise RuntimeError(f"No usable motions found for split {args.split}")
    return torch.from_numpy(np.stack(motions, axis=0)), torch.tensor(lengths, dtype=torch.long), names


def main():
    args = parse_args()
    kv_root = Path(args.kv_root).expanduser().resolve()
    mean, std, stats_summary = load_stats(args, kv_root)
    motions, lengths, names = load_motions(args, mean, std)
    tokenizer = PartVQTokenizer(
        kv_root=str(kv_root),
        checkpoint_path=args.vq_checkpoint or str(kv_root / "checkpoints" / "vqvae" / "net_best_fid.pth"),
        partition_path=args.vq_partition or str(kv_root / "checkpoints" / "vqvae" / "skeleton_partition.json"),
        device=torch.device(args.device),
    )
    summary = tokenizer.verify_contract(
        motion=motions.to(tokenizer.device),
        lengths=lengths.to(tokenizer.device),
        max_samples=len(names),
    )
    summary["stats"] = stats_summary
    summary["motion_ids"] = names
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
