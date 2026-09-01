#!/usr/bin/env python3
"""J5: the geometry correlation measured against DECODED prototypes.

The metric-alignment diagnostic in the paper builds each code's prototype from
the normalized motion features observed wherever that code fires -- an
ENCODER-side quantity.  The abstract and conclusion, however, claim that code
distances track the distances between the movements those codes "decode to".

This tool measures that second quantity directly.  For each group p it takes a
number of real contexts, and for each context substitutes every code k of that
group at one valid step, decodes the whole sequence through the frozen
decoder, and reads back the local window at the substituted step restricted to
that group's channels.  Averaging over contexts gives a decoder-side prototype
per code.  The Spearman correlation between code-embedding distance and
decoder-side prototype distance is then the quantity the abstract asserts.

All substitutions are batched: one decode call per (part, context) covers the
whole codebook.  Encoding, part segmentation and active-code selection are
imported unchanged from the existing diagnostics.
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
DEFAULT_OUT = Path("eval_results/diag_20260831/decoder_side_rho.json")


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
    # J5-specific
    p.add_argument("--contexts", type=int, default=16,
                   help="Real contexts averaged into each decoder-side prototype.")
    p.add_argument("--decode-chunk", type=int, default=64,
                   help="Codes decoded per forward pass.")
    return p.parse_args()


def decode_batch(model, emb: torch.Tensor) -> torch.Tensor:
    """emb: [B, T, P, d] -> decoded motion [B, F, feature_dim]."""
    parts = [emb[:, :, i].permute(0, 2, 1).contiguous() for i in range(emb.shape[2])]
    x_quantized = torch.cat(parts, dim=1)
    return model.vqvae.postprocess(model.vqvae.decoder(x_quantized))


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
    encoded, proto_sums_by_space, counts, _meta = diag.encode_dataset(
        model, args.kv_root, args.data_root, mean, std, split_ids,
        partition, part_seg, args, device, rng,
    )
    codebooks = diag.codebooks_numpy(model)
    num_parts = len(codebooks)
    n_motions = encoded.ids.shape[0]

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for p in range(num_parts):
            emb_p = codebooks[p]
            K = emb_p.shape[0]
            active = np.flatnonzero(counts[p] >= args.min_code_count)
            if len(active) < 4:
                rows.append({"part": p, "part_name": diag.part_name(p),
                             "active_codes": int(len(active)), "note": "too few active codes"})
                continue

            proto_sum = None
            n_ctx_used = 0
            for _ in range(args.contexts):
                n = int(rng.integers(n_motions))
                valid_ts = np.flatnonzero(encoded.latent_valid[n])
                if len(valid_ts) == 0:
                    continue
                t = int(rng.choice(valid_ts))
                ids = encoded.ids[n:n + 1].astype(np.int64)
                base = np.stack([codebooks[q][ids[0, :, q]] for q in range(num_parts)], axis=1)[None]
                frame_lo, frame_hi = t * 4, min(t * 4 + 4, int(encoded.lengths[n]))
                if frame_hi <= frame_lo:
                    continue

                per_code = np.empty((K, (frame_hi - frame_lo) * len(part_seg[p])), dtype=np.float64)
                for lo in range(0, K, args.decode_chunk):
                    hi = min(lo + args.decode_chunk, K)
                    batch = np.repeat(base, hi - lo, axis=0)
                    batch[:, t, p] = emb_p[lo:hi]
                    out = decode_batch(model, torch.from_numpy(batch).float().to(device))
                    win = out[:, frame_lo:frame_hi][:, :, part_seg[p]]
                    per_code[lo:hi] = win.reshape(hi - lo, -1).double().cpu().numpy()

                proto_sum = per_code if proto_sum is None else proto_sum + per_code
                n_ctx_used += 1

            if n_ctx_used == 0:
                rows.append({"part": p, "part_name": diag.part_name(p), "note": "no usable context"})
                continue
            proto_dec = proto_sum[active] / n_ctx_used

            pairs = diag.sample_pairs(len(active), args.max_pairs, rng)
            code_d = diag.pairwise_dist_rows(emb_p[active].astype(np.float64), pairs)
            dec_d = diag.pairwise_dist_rows(proto_dec, pairs)
            rho_dec = diag.spearman(code_d, dec_d)
            boot_mean, ci_lo, ci_hi = diag.bootstrap_rank_ci(code_d, dec_d, args.bootstrap, rng)

            # encoder-side reference on exactly the same code set and pairs
            proto_enc = proto_sums_by_space["feature"][p][active] / counts[p, active, None].clip(min=1)
            rho_enc = diag.spearman(code_d, diag.pairwise_dist_rows(proto_enc, pairs))

            rows.append({
                "part": p, "part_name": diag.part_name(p),
                "active_codes": int(len(active)), "contexts": int(n_ctx_used),
                "pairs": int(len(pairs)),
                "spearman_decoder_side": float(rho_dec),
                "spearman_decoder_ci_low": float(ci_lo),
                "spearman_decoder_ci_high": float(ci_hi),
                "spearman_encoder_side": float(rho_enc),
            })
            print("[part {}] {:20s} decoder rho={:.4f} [{:.3f},{:.3f}]  encoder rho={:.4f}".format(
                p, diag.part_name(p), rho_dec, ci_lo, ci_hi, rho_enc), flush=True)

    scored = [r for r in rows if "spearman_decoder_side" in r]
    out = {
        "split": args.split, "seed": args.seed, "checkpoint": str(checkpoint_path),
        "contexts": args.contexts,
        "mean_spearman_decoder_side": float(np.mean([r["spearman_decoder_side"] for r in scored])),
        "mean_spearman_encoder_side": float(np.mean([r["spearman_encoder_side"] for r in scored])),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n{:20s} {:6s} {:>10s} {:>18s} {:>10s}".format("group", "codes", "rho_dec", "95% CI", "rho_enc"))
    print("-" * 70)
    for r in rows:
        if "spearman_decoder_side" not in r:
            continue
        print("{:20s} {:6d} {:10.4f}   [{:.3f},{:.3f}] {:10.4f}".format(
            r["part_name"], r["active_codes"], r["spearman_decoder_side"],
            r["spearman_decoder_ci_low"], r["spearman_decoder_ci_high"],
            r["spearman_encoder_side"]))
    print("-" * 70)
    print("{:20s} {:6s} {:10.4f} {:20s} {:10.4f}".format(
        "MEAN", "", out["mean_spearman_decoder_side"], "", out["mean_spearman_encoder_side"]))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
