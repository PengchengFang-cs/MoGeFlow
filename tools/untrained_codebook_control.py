#!/usr/bin/env python3
"""Untrained-tokenizer null for the codebook-geometry diagnostic.

The shuffled control only rules out usage statistics: it destroys any
correspondence, so beating it is easy.  The sharper question a reviewer asks is
whether "code-embedding distance tracks motion-prototype distance" is a
*learned* property or an automatic consequence of nearest-neighbour assignment
under any continuous encoder.

This tool answers it by re-initialising the tokenizer's encoder and codebooks at
random, re-encoding the same motions with that untrained tokenizer, building
prototypes from its own assignments, and running the identical Spearman
diagnostic.  A high rho there would mean the property is automatic; a collapse
means it is learned.

Everything except the randomisation is imported from the existing diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import motion_code_geometry_diagnostics as diag  # noqa: E402
import motion_code_geometry_diagnostics_formal as formal  # noqa: E402

DEFAULT_KV_ROOT = Path("/scratch/pf2m24/projects/Umdd/KV-Control")
DEFAULT_DATA_ROOT = Path("/scratch/pf2m24/data/HumanML3D/HumanML3D")
DEFAULT_OUT = Path("eval_results/diag_20260820/untrained_codebook.json")


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
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--nn-k", type=int, nargs="+", default=[1])
    p.add_argument("--replacement-samples", type=int, default=0)
    p.add_argument("--crop", default="start", choices=["start", "center", "random"])
    p.add_argument("--prototype-spaces", nargs="+", default=["feature"])
    p.add_argument("--confound-controls", action="store_true")
    p.add_argument("--error-cost", action="store_true")
    p.add_argument("--skip-replacement", action="store_true", default=True)
    p.add_argument("--max-error-pairs", type=int, default=20000)
    return p.parse_args()


def randomize_tokenizer(model, seed: int) -> Dict[str, int]:
    """Re-initialise every encoder and quantizer parameter; leave shapes intact."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    touched = {"modules": 0, "tensors": 0}
    vq = model.vqvae
    targets = []
    for attr in ("encoders", "encoder", "quantizers"):
        if hasattr(vq, attr):
            targets.append(getattr(vq, attr))
    for target in targets:
        modules = target.modules() if isinstance(target, nn.Module) else []
        for m in modules:
            reset = getattr(m, "reset_parameters", None)
            if callable(reset):
                reset()
                touched["modules"] += 1
        params = target.parameters() if isinstance(target, nn.Module) else []
        for prm in params:
            with torch.no_grad():
                new = torch.empty(prm.shape, dtype=torch.float32).normal_(0.0, 0.02, generator=g)
                prm.copy_(new.to(prm.device, prm.dtype))
            touched["tensors"] += 1
        buffers = target.buffers() if isinstance(target, nn.Module) else []
        for buf in buffers:
            if buf.dtype.is_floating_point and buf.dim() >= 2:
                with torch.no_grad():
                    new = torch.empty(buf.shape, dtype=torch.float32).normal_(0.0, 1.0, generator=g)
                    buf.copy_(new.to(buf.device, buf.dtype))
                touched["tensors"] += 1
    return touched


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

    touched = randomize_tokenizer(model, args.seed)
    print(f"[randomize] reset {touched['modules']} modules, overwrote {touched['tensors']} tensors")

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
            if len(active) < 4:
                rows.append({"part": p, "part_name": diag.part_name(p),
                             "active_codes": int(len(active)), "note": "too few active codes"})
                continue
            proto = proto_sums[p][active] / counts[p, active, None].clip(min=1)
            pairs = diag.sample_pairs(len(active), args.max_pairs, rng)
            motion_d = diag.pairwise_dist_rows(proto, pairs)
            code_d = diag.pairwise_dist_rows(emb[active].astype(np.float64), pairs)
            boot_mean, ci_low, ci_high = diag.bootstrap_rank_ci(code_d, motion_d, args.bootstrap, rng)
            rows.append({
                "prototype_space": space, "part": p, "part_name": diag.part_name(p),
                "active_codes": int(len(active)), "pairs": int(len(pairs)),
                "spearman": diag.spearman(code_d, motion_d),
                "spearman_ci_low": ci_low, "spearman_ci_high": ci_high,
            })

    scored = [r for r in rows if "spearman" in r]
    mean_rho = float(np.mean([r["spearman"] for r in scored])) if scored else float("nan")
    out = {"split": args.split, "seed": args.seed, "untrained": True,
           "checkpoint_shape_source": str(checkpoint_path),
           "mean_spearman": mean_rho, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'group':20s} {'codes':>6s} {'rho':>8s}   CI")
    print("-" * 56)
    for r in rows:
        if "spearman" in r:
            print("{:20s} {:6d} {:8.4f}   [{:.3f},{:.3f}]".format(
                r["part_name"], r["active_codes"], r["spearman"],
                r["spearman_ci_low"], r["spearman_ci_high"]))
        else:
            print("{:20s} {:6d}   {}".format(r["part_name"], r["active_codes"], r["note"]))
    print("-" * 56)
    print(f"{'MEAN':20s} {'':6s} {mean_rho:8.4f}")
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
