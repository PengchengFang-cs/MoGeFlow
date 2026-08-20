#!/usr/bin/env python3
"""Norm-matched off-lattice control for the code-replacement diagnostic.

The replacement panel in the paper substitutes a code with a near / mid / far
codebook neighbour and reports how far the frozen decoder's output moves.  A
Lipschitz decoder would produce the same monotone pattern for *any* input
perturbation, so that panel alone cannot separate "codebook geometry" from
"the decoder is smooth".

This tool adds the discriminating control: for every substitution it also
perturbs the same code embedding by a random direction of the *same* L2 norm,
which lands off the lattice, and decodes that instead.  If moving along the
codebook is gentler than moving an equal distance in a random direction, the
lattice carries structure the decoder respects; if the two match, the panel was
only measuring decoder smoothness.

Encoding, prototype construction and the near/mid/far selection are imported
from ``motion_code_geometry_diagnostics``; nothing there is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import motion_code_geometry_diagnostics as diag  # noqa: E402
import motion_code_geometry_diagnostics_formal as formal  # noqa: E402

DEFAULT_KV_ROOT = Path("/scratch/pf2m24/projects/Umdd/KV-Control")
DEFAULT_DATA_ROOT = Path("/scratch/pf2m24/data/HumanML3D/HumanML3D")
DEFAULT_OUT = Path("eval_results/diag_20260820/offlattice_replacement.json")


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
    p.add_argument("--nn-k", type=int, nargs="+", default=[1])
    p.add_argument("--replacement-samples", type=int, default=128)
    p.add_argument("--crop", default="start", choices=["start", "center", "random"])
    p.add_argument("--prototype-spaces", nargs="+", default=["feature"])
    p.add_argument("--confound-controls", action="store_true")
    p.add_argument("--error-cost", action="store_true")
    p.add_argument("--skip-replacement", action="store_true")
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--max-error-pairs", type=int, default=20000)
    return p.parse_args()


def decode_from_embeddings(model, emb: torch.Tensor) -> np.ndarray:
    """emb: [1, T, P, d] -> decoded motion [T_frames, feature_dim] as numpy."""
    parts = [emb[:, :, i].permute(0, 2, 1).contiguous() for i in range(emb.shape[2])]
    x_quantized = torch.cat(parts, dim=1)
    x_decoder = model.vqvae.decoder(x_quantized)
    out = model.vqvae.postprocess(x_decoder)
    return out.detach().cpu().numpy()[0]


def unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.normal(size=dim)
    return v / (np.linalg.norm(v) + 1e-12)


def mean_ci(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "ci": float("nan"), "n": 0}
    mean = float(arr.mean())
    ci = 1.96 * float(arr.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "ci": ci, "n": n}


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
    part_seg = partition["partSeg"]
    encoded, _proto, _counts, _meta = diag.encode_dataset(
        model, args.kv_root, args.data_root, mean, std, split_ids,
        partition, part_seg, args, device, rng,
    )
    codebooks = diag.codebooks_numpy(model)

    records: List[Dict[str, object]] = []
    n_total = encoded.ids.shape[0]
    sample_idx = rng.choice(n_total, size=min(args.replacement_samples, n_total), replace=False)

    with torch.no_grad():
        for n in sample_idx:
            ids = encoded.ids[n: n + 1].astype(np.int64)
            valid_ts = np.flatnonzero(encoded.latent_valid[n])
            if len(valid_ts) == 0:
                continue
            base_emb = np.stack(
                [codebooks[p][ids[0, :, p]] for p in range(ids.shape[-1])], axis=1
            )[None]  # [1, T, P, d]
            base_t = torch.from_numpy(base_emb).float().to(device)
            original = decode_from_embeddings(model, base_t)
            valid_frames = np.arange(encoded.motions.shape[1]) < encoded.lengths[n]

            for p in range(ids.shape[-1]):
                t = int(rng.choice(valid_ts))
                old_code = int(ids[0, t, p])
                e_old = codebooks[p][old_code]
                repl = diag.choose_near_mid_far(codebooks[p], old_code)
                frame_start, frame_end = t * 4, min(t * 4 + 4, int(encoded.lengths[n]))
                local_frames = np.arange(frame_start, frame_end)

                for category, new_code in repl.items():
                    e_new = codebooks[p][int(new_code)]
                    dist = float(np.linalg.norm(e_new - e_old))
                    variants = {
                        "lattice": e_new,
                        "random_direction": e_old + dist * unit_vector(rng, e_old.shape[0]).astype(e_old.dtype),
                    }
                    for variant, e_sub in variants.items():
                        emb = base_emb.copy()
                        emb[0, t, p] = e_sub
                        mod = decode_from_embeddings(model, torch.from_numpy(emb).float().to(device))
                        delta = np.abs(mod - original)
                        whole = float(delta[valid_frames].mean())
                        target = float(delta[np.ix_(valid_frames, part_seg[p])].mean())
                        local = (float(delta[np.ix_(local_frames, part_seg[p])].mean())
                                 if len(local_frames) else float("nan"))
                        records.append({
                            "motion": int(n), "part": int(p), "step": t,
                            "category": category, "variant": variant,
                            "code_distance": dist,
                            "whole": whole, "target": target, "local": local,
                        })

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for category in ("near", "mid", "far"):
        summary[category] = {}
        for variant in ("lattice", "random_direction"):
            sub = [r for r in records if r["category"] == category and r["variant"] == variant]
            summary[category][variant] = {
                "code_distance": mean_ci([r["code_distance"] for r in sub])["mean"],
                "whole": mean_ci([r["whole"] for r in sub]),
                "target": mean_ci([r["target"] for r in sub]),
                "local": mean_ci([r["local"] for r in sub]),
            }

    out = {
        "split": args.split, "seed": args.seed, "num_motions": int(len(sample_idx)),
        "replacements": len(records), "checkpoint": str(checkpoint_path),
        "summary": summary, "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'cat':5s} {'variant':17s} {'code dist':>10s} {'whole':>18s} {'target':>18s} {'local':>18s}")
    print("-" * 92)
    for category in ("near", "mid", "far"):
        for variant in ("lattice", "random_direction"):
            e = summary[category][variant]
            print("{:5s} {:17s} {:10.2f} {:9.5f}±{:.5f} {:9.5f}±{:.5f} {:9.5f}±{:.5f}".format(
                category, variant, e["code_distance"],
                e["whole"]["mean"], e["whole"]["ci"],
                e["target"]["mean"], e["target"]["ci"],
                e["local"]["mean"], e["local"]["ci"]))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
