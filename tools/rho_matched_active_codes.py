#!/usr/bin/env python3
"""J2: recompute the geometry correlation with the active-code count matched.

The untrained-tokenizer control reports rho 0.24 against the trained
tokenizer's 0.82, but it also keeps only ~37 active codes per group against
113.  Spearman is computed over code pairs, so 37 codes (666 pairs) and 113
codes (6328 pairs) are not directly comparable: part of the gap could be
support size rather than training.

This tool removes that confound.  It runs the identical diagnostic on the
TRAINED tokenizer, but subsamples the active codes of each group down to a
target count (default 37) before computing the correlation, and repeats the
draw many times to report the spread.  If the trained tokenizer still scores
far above 0.24 at matched support, the control stands.

Everything except the subsampling is imported from the existing diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import motion_code_geometry_diagnostics as diag  # noqa: E402
import motion_code_geometry_diagnostics_formal as formal  # noqa: E402

DEFAULT_KV_ROOT = Path("/scratch/pf2m24/projects/Umdd/KV-Control")
DEFAULT_DATA_ROOT = Path("/scratch/pf2m24/data/HumanML3D/HumanML3D")
DEFAULT_OUT = Path("eval_results/diag_20260831/rho_matched_active.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kv-root", type=Path, default=DEFAULT_KV_ROOT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vq-checkpoint", type=Path, default=None)
    p.add_argument("--partition-file", type=Path, default=None)
    p.add_argument("--mean-path", type=Path, default=None)
    p.add_argument("--std-path", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "train_val", "all"])
    p.add_argument("--max-motions", type=int, default=512)
    p.add_argument("--motion-length", type=int, default=196)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--min-code-count", type=int, default=20)
    p.add_argument("--max-pairs", type=int, default=20000)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--nn-k", type=int, nargs="+", default=[1])
    p.add_argument("--replacement-samples", type=int, default=0)
    p.add_argument("--crop", default="start", choices=["start", "center", "random"])
    p.add_argument("--prototype-spaces", nargs="+", default=["feature"])
    p.add_argument("--confound-controls", action="store_true")
    p.add_argument("--error-cost", action="store_true")
    p.add_argument("--skip-replacement", action="store_true", default=True)
    p.add_argument("--max-error-pairs", type=int, default=20000)
    # J2-specific
    p.add_argument("--target-active", type=int, default=37,
                   help="Active codes to subsample each group down to (untrained control had 37).")
    p.add_argument("--draws", type=int, default=200, help="Random subsampling draws per group.")
    return p.parse_args()


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
    diag.PART_NAMES = list(formal.formal_part_names(partition))
    diag.recover_joints_np = formal.make_recover_joints_np(
        int(partition.get("joints_num", 22 if feature_dim == 263 else 21))
    )

    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    std = np.where(std == 0, 1.0, std)

    split_ids = diag.load_split_ids(args.data_root, args.split)
    encoded, proto_sums_by_space, counts, _meta = diag.encode_dataset(
        model, args.kv_root, args.data_root, mean, std, split_ids,
        partition, partition["partSeg"], args, device, rng,
    )
    del encoded
    codebooks = diag.codebooks_numpy(model)

    rows: List[Dict[str, object]] = []
    for space, proto_sums in proto_sums_by_space.items():
        for p, emb in enumerate(codebooks):
            active = np.flatnonzero(counts[p] >= args.min_code_count)
            proto_full = proto_sums[p][active] / counts[p, active, None].clip(min=1)
            emb_full = emb[active].astype(np.float64)

            # full-support reference (this is the number in the paper table)
            pairs = diag.sample_pairs(len(active), args.max_pairs, rng)
            rho_full = diag.spearman(
                diag.pairwise_dist_rows(emb_full, pairs),
                diag.pairwise_dist_rows(proto_full, pairs),
            )

            # matched-support draws
            n_take = min(args.target_active, len(active))
            draws = []
            for _ in range(args.draws):
                sel = rng.choice(len(active), size=n_take, replace=False)
                sub_pairs = diag.sample_pairs(n_take, args.max_pairs, rng)
                draws.append(diag.spearman(
                    diag.pairwise_dist_rows(emb_full[sel], sub_pairs),
                    diag.pairwise_dist_rows(proto_full[sel], sub_pairs),
                ))
            draws = np.asarray(draws, dtype=np.float64)
            rows.append({
                "prototype_space": space, "part": p, "part_name": diag.part_name(p),
                "active_codes": int(len(active)), "subsampled_to": int(n_take),
                "draws": int(args.draws),
                "spearman_full_support": float(rho_full),
                "spearman_matched_mean": float(draws.mean()),
                "spearman_matched_std": float(draws.std(ddof=1)) if len(draws) > 1 else 0.0,
                "spearman_matched_p2.5": float(np.percentile(draws, 2.5)),
                "spearman_matched_p97.5": float(np.percentile(draws, 97.5)),
            })

    scored = [r for r in rows if "spearman_full_support" in r]
    out = {
        "split": args.split, "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "target_active": args.target_active, "draws": args.draws,
        "mean_spearman_full_support": float(np.mean([r["spearman_full_support"] for r in scored])),
        "mean_spearman_matched": float(np.mean([r["spearman_matched_mean"] for r in scored])),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'group':20s} {'act':>5s} {'->n':>5s} {'rho_full':>9s} {'rho_matched':>12s}   95% range")
    print("-" * 78)
    for r in rows:
        print("{:20s} {:5d} {:5d} {:9.4f} {:12.4f}   [{:.3f},{:.3f}]".format(
            r["part_name"], r["active_codes"], r["subsampled_to"],
            r["spearman_full_support"], r["spearman_matched_mean"],
            r["spearman_matched_p2.5"], r["spearman_matched_p97.5"]))
    print("-" * 78)
    print("{:20s} {:5s} {:5s} {:9.4f} {:12.4f}".format(
        "MEAN", "", "", out["mean_spearman_full_support"], out["mean_spearman_matched"]))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
