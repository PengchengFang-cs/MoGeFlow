#!/usr/bin/env python3
"""E2: whitened per-group Spearman rho between codebook and prototype distances.

Recomputes the Table-3 style per-group Spearman correlation between code
embedding distances and decoded-prototype distances, side by side under

  raw:      D_p(k, k') = || e_pk - e_pk' ||_2
  whitened: D_p(k, k') = || (e_pk - e_pk') / sigma_p ||_2

where sigma_p is the per-dimension codebook std over all codes of part p --
exactly the statistic ``latent_norm_mode=codebook`` uses in
``motion_code_flow.MotionCodeFlow._init_latent_stats``
(``codebooks.std(dim=1, unbiased=False).clamp_min(latent_norm_eps)``; the mean
``mu_p`` cancels in pairwise differences and is recorded for reference only).

Prototype construction, encoding, pair sampling, Spearman, and bootstrap CI
machinery are reused by import from ``motion_code_geometry_diagnostics`` and
its formal loader ``motion_code_geometry_diagnostics_formal``; no diagnostics
code is duplicated or modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(TOOLS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import motion_code_geometry_diagnostics as diag
import motion_code_geometry_diagnostics_formal as formal


DEFAULT_KV_ROOT = Path("/scratch/pf2m24/projects/Umdd/KV-Control")
DEFAULT_DATA_ROOT = REPO_ROOT / "dataset/HumanML3D"
DEFAULT_OUT = REPO_ROOT / "eval_results/diag_20260809/whitened_rho.json"
LATENT_NORM_EPS = 1e-6  # matches MotionCodeFlowConfig.latent_norm_eps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw vs whitened codebook-vs-prototype Spearman rho per part group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kv-root", type=Path, default=DEFAULT_KV_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vq-checkpoint", type=Path, default=None, help="Defaults to the formal overlap tokenizer resolution.")
    parser.add_argument("--partition-file", type=Path, default=None)
    parser.add_argument("--mean-path", type=Path, default=None)
    parser.add_argument("--std-path", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "train_val", "all"])
    parser.add_argument("--max-motions", type=int, default=512)
    parser.add_argument("--motion-length", type=int, default=196)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-code-count", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for Spearman CI; 0 disables.")
    parser.add_argument("--crop", default="start", choices=["start", "center", "random"])
    parser.add_argument(
        "--prototype-spaces",
        nargs="+",
        default=["feature"],
        choices=["feature", "joint_pos", "joint_pos_vel"],
    )
    return parser.parse_args()


def codebook_whitening_stats(codebook: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-dimension mu/sigma over the FULL codebook, as latent_norm_mode=codebook."""
    mu = codebook.mean(axis=0)
    sigma = codebook.std(axis=0)  # ddof=0 == torch std(unbiased=False)
    sigma = np.clip(sigma, LATENT_NORM_EPS, None)
    return {"mu": mu.astype(np.float64), "sigma": sigma.astype(np.float64)}


def main() -> None:
    args = parse_args()
    rng = diag.seed_everything(args.seed)
    args.kv_root = args.kv_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()

    checkpoint_path = formal.resolve_vq_checkpoint(args.kv_root, args.vq_checkpoint)
    partition_path = formal.resolve_partition(args.kv_root, args.partition_file)
    mean_path, std_path = formal.resolve_stats(args.kv_root, args.data_root, args.mean_path, args.std_path)

    device = torch.device(args.device)
    model, partition, dataname, feature_dim = formal.load_formal_vq_model(
        args.kv_root, checkpoint_path, partition_path, device
    )
    part_names = formal.formal_part_names(partition)
    diag.PART_NAMES = list(part_names)
    diag.recover_joints_np = formal.make_recover_joints_np(
        int(partition.get("joints_num", 22 if feature_dim == 263 else 21))
    )

    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    std = np.where(std == 0, 1.0, std)

    split_ids = diag.load_split_ids(args.data_root, args.split)
    encoded, proto_sums_by_space, counts, meta = diag.encode_dataset(
        model,
        args.kv_root,
        args.data_root,
        mean,
        std,
        split_ids,
        partition,
        partition["partSeg"],
        args,
        device,
        rng,
    )
    del encoded
    codebooks = diag.codebooks_numpy(model)

    rows: List[Dict[str, object]] = []
    for space, proto_sums in proto_sums_by_space.items():
        for p, emb in enumerate(codebooks):
            active = np.flatnonzero(counts[p] >= args.min_code_count)
            if len(active) < 4:
                continue
            proto = proto_sums[p][active] / counts[p, active, None].clip(min=1)
            stats = codebook_whitening_stats(emb)
            learned_raw = emb[active].astype(np.float64)
            learned_white = learned_raw / stats["sigma"][None, :]
            perm = rng.permutation(len(learned_raw))

            pairs = diag.sample_pairs(len(active), args.max_pairs, rng)
            motion_d = diag.pairwise_dist_rows(proto, pairs)

            metrics: Dict[str, Dict[str, float]] = {}
            for metric_name, embedding in (
                ("raw", learned_raw),
                ("whitened", learned_white),
                ("raw_shuffled", learned_raw[perm]),
                ("whitened_shuffled", learned_white[perm]),
            ):
                code_d = diag.pairwise_dist_rows(embedding, pairs)
                boot_mean, ci_low, ci_high = diag.bootstrap_rank_ci(code_d, motion_d, args.bootstrap, rng)
                metrics[metric_name] = {
                    "spearman": diag.spearman(code_d, motion_d),
                    "pearson": diag.pearson(code_d, motion_d),
                    "spearman_boot_mean": boot_mean,
                    "spearman_ci_low": ci_low,
                    "spearman_ci_high": ci_high,
                }

            rows.append(
                {
                    "prototype_space": space,
                    "part": p,
                    "part_name": diag.part_name(p),
                    "active_codes": int(len(active)),
                    "pairs": int(len(pairs)),
                    "sigma_min": float(stats["sigma"].min()),
                    "sigma_mean": float(stats["sigma"].mean()),
                    "sigma_max": float(stats["sigma"].max()),
                    "mu_abs_mean": float(np.abs(stats["mu"]).mean()),
                    "metrics": metrics,
                }
            )

    if not rows:
        raise RuntimeError("No part groups passed the min-code-count threshold")

    mean_rows: Dict[str, Dict[str, float]] = {}
    for space in sorted({row["prototype_space"] for row in rows}):
        space_rows = [row for row in rows if row["prototype_space"] == space]
        mean_rows[space] = {
            "spearman_raw_mean": float(np.mean([r["metrics"]["raw"]["spearman"] for r in space_rows])),
            "spearman_whitened_mean": float(np.mean([r["metrics"]["whitened"]["spearman"] for r in space_rows])),
            "spearman_raw_shuffled_mean": float(np.mean([r["metrics"]["raw_shuffled"]["spearman"] for r in space_rows])),
            "spearman_whitened_shuffled_mean": float(
                np.mean([r["metrics"]["whitened_shuffled"]["spearman"] for r in space_rows])
            ),
        }

    payload = {
        **meta,
        "kv_root": str(args.kv_root),
        "data_root": str(args.data_root),
        "vq_checkpoint": str(checkpoint_path),
        "partition_file": str(partition_path),
        "mean_path": str(mean_path),
        "std_path": str(std_path),
        "split": args.split,
        "seed": int(args.seed),
        "max_motions": int(args.max_motions),
        "min_code_count": int(args.min_code_count),
        "max_pairs": int(args.max_pairs),
        "bootstrap": int(args.bootstrap),
        "latent_norm_eps": LATENT_NORM_EPS,
        "whitened_metric": "D_p(k,k') = ||(e_pk - e_pk') / sigma_p||_2, sigma_p = full-codebook per-dim std (unbiased=False)",
        "part_names": list(part_names),
        "rows": rows,
        "mean_over_parts": mean_rows,
    }

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=diag.json_default) + "\n",
        encoding="utf-8",
    )

    header = f"{'space':<10} {'group':<20} {'rho_raw':>8} {'CI_raw':>18} {'rho_wht':>8} {'CI_wht':>18} {'shuf_raw':>9} {'shuf_wht':>9}"
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        m = row["metrics"]
        ci_raw = f"[{m['raw']['spearman_ci_low']:.3f},{m['raw']['spearman_ci_high']:.3f}]"
        ci_wht = f"[{m['whitened']['spearman_ci_low']:.3f},{m['whitened']['spearman_ci_high']:.3f}]"
        print(
            f"{row['prototype_space']:<10} {row['part_name']:<20} "
            f"{m['raw']['spearman']:>8.4f} {ci_raw:>18} "
            f"{m['whitened']['spearman']:>8.4f} {ci_wht:>18} "
            f"{m['raw_shuffled']['spearman']:>9.4f} {m['whitened_shuffled']['spearman']:>9.4f}"
        )
    print("-" * len(header))
    for space, summary in mean_rows.items():
        print(
            f"{space:<10} {'MEAN':<20} {summary['spearman_raw_mean']:>8.4f} {'':>18} "
            f"{summary['spearman_whitened_mean']:>8.4f} {'':>18} "
            f"{summary['spearman_raw_shuffled_mean']:>9.4f} {summary['spearman_whitened_shuffled_mean']:>9.4f}"
        )
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
