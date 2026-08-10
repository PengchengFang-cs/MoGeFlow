#!/usr/bin/env python3
"""Offline RVQ token export for Graph-CodeFlow (handoff/20260609_graph_codeflow_
rvq_backbone_plan.md §5.1 + M1).

Runs the FROZEN Graph-VQVAE tokenizer over the AnyTop L5 dataset and caches, per
clip, the post-RVQ targets + graph metadata + caption refs the CodeFlow backbone
needs so training never re-runs the encoder online:

  z_q                         [T_lat,C,D] fp16   (RAW frozen RVQ z_q target)
  indices                     [T_lat,C,Q] int16  (-1 on padded tokens)
  token_mask                  [T_lat,C]   bool
  coarse_mask                 [C]         bool
  frame_mask_lat              [T_lat]     bool
  pooled_adjacency            [C,C]       fp16
  pooled_geodesic             [C,C]       fp16  (+inf -> large sentinel on save)
  pooled_skeleton_embeddings  [C,D]       fp16
  assignment                  [J,C]       fp16
  parent_indices, rest_offsets, anytop_mean/std, joint_mask  (decode metadata)
  caption_emb [768] + caption_token_emb [L,768] + caption_token_mask [L] + text

HARD export invariants (review verdict gaps #3/#4 + M1 gates):
  - tokenizer.eval()  (training-mode RVQ has quantizer-dropout that truncates depth)
  - allow_collectives=False  (single-process; no cross-rank collectives)
  - FULL Q (eval => no dropout)  asserted: every VALID token has all-Q indices >= 0
  - captions ON (load_captions=True + caption caches), unlike VQ training
  - strict shape audit + padded IDs exactly -1 + valid IDs in [0,K-1]
  - ids_to_embeddings(indices) ~= cached z_q within tol  (per-clip)
  - train/val split mirrors the VQVAE source split (val_frac/seed from the ckpt)

For the SMOKE, pass --max_clips to export only a few clips to /tmp; the FULL export
waits for the ep300 frozen ckpt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.vq_model import GraphVQTokenizer

# Sentinel for +inf in pooled_geodesic on disk (unreachable pairs). The CodeFlow
# loader maps it back to +inf before feeding GraphAttentionBlock. Chosen well
# above any real hop count (<= C-1 <= 49) and within fp16 range.
GEO_INF_SENTINEL = 30000.0


def load_frozen_tokenizer(ckpt_path: str, dev: torch.device):
    """Rebuild GraphVQTokenizer from a train_graph_vqvae.py ckpt (saved `args`),
    load weights strict=True, eval(). Mirrors animate_vqvae_recon.load_vq_tokenizer."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = ck.get("args", {})
    if not ta:
        raise SystemExit(f"export: ckpt {ckpt_path} missing 'args' "
                         f"(not from train_graph_vqvae.py?)")
    model = GraphVQTokenizer(
        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
        n_graph_layers=ta["n_graph_layers"],
        n_enc_temporal_layers=ta["n_enc_temporal_layers"],
        n_pre_vq_layers=ta["n_pre_vq_layers"],
        n_post_vq_layers=ta["n_post_vq_layers"],
        n_cross_layers=ta["n_cross_layers"],
        n_dec_temporal_layers=ta["n_dec_temporal_layers"],
        max_coarse=ta["max_coarse"], temporal_stride=ta["temporal_stride"],
        temporal_kernel=ta["temporal_kernel"], dropout=ta["dropout"],
        code_dim=ta["code_dim"], num_codes=ta["num_codes"],
        num_quantizers=ta["num_quantizers"], ema_mu=ta["ema_mu"],
        quantize_dropout_prob=ta["quantize_dropout_prob"],
        dead_code_threshold=ta["dead_code_threshold"],
    ).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model, ta, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen_vqvae_ckpt", required=True)
    ap.add_argument("--out", required=True, help="output dir for per-split token caches")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--anytop_root", type=str, default=None,
                    help="defaults to the ckpt's training root")
    ap.add_argument("--caption_emb_cache", type=str,
                    default="data/anytop_caption_t5_cleanL5_multi.npz")
    ap.add_argument("--caption_token_cache", type=str,
                    default="data/anytop_caption_t5_cleanL5_multi",
                    help="prefix for .tokens.npy/.token_mask.npy/.keys.json")
    ap.add_argument("--caption_token_max_len", type=int, default=64)
    ap.add_argument("--num_frames", type=int, default=None,
                    help="override the frozen tokenizer ckpt's max_frames; None = "
                         "use ckpt's max_frames. Full-length backbone training uses "
                         "300 -> T_lat up to 75")
    ap.add_argument("--min_text_coverage", type=float, default=0.99,
                    help="PREFLIGHT: minimum fraction of exported clips that must "
                         "carry non-empty text (non-zero caption_emb AND "
                         "caption_token_mask.sum()>0). Default 0.99 fails loud on a "
                         "text-less cache; lower it explicitly for a deliberate "
                         "unconditional export.")
    ap.add_argument("--max_clips", type=int, default=0,
                    help="SMOKE: cap clips per split (0 = all)")
    ap.add_argument("--shard_idx", type=int, default=0,
                    help="SHARD: this process exports the disjoint subset of clips "
                         "with (dataset_index %% num_shards == shard_idx)")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="SHARD: total parallel processes. >1 enables sharded mode: "
                         "each shard writes index_shard{idx:03d}.jsonl (not the shared "
                         "index.jsonl/manifest.json; the merge step writes those). "
                         "1 (default) = single-process path, unchanged.")
    ap.add_argument("--amp_dtype", choices=["fp32", "bf16"], default=None,
                    help="autocast dtype; defaults to the ckpt's training amp_dtype")
    ap.add_argument("--identity_tol", type=float, default=1e-2,
                    help="max |ids_to_embeddings(indices) - z_q| tolerated on valid "
                         "tokens (fp16 storage round-trip; the fp32 audit before "
                         "casting must be ~1e-5)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.num_shards < 1:
        raise SystemExit(f"export: --num_shards must be >= 1 (got {args.num_shards})")
    if not (0 <= args.shard_idx < args.num_shards):
        raise SystemExit(f"export: --shard_idx must be in [0, num_shards) "
                         f"(got shard_idx={args.shard_idx}, num_shards={args.num_shards})")
    sharded = args.num_shards > 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable -> CPU")
        args.device = "cpu"
    dev = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ta, ck = load_frozen_tokenizer(args.frozen_vqvae_ckpt, dev)
    K = ta["num_codes"]
    Q = ta["num_quantizers"]
    D = ta["d_model"]
    print(f"Frozen tokenizer: d_model={D} max_coarse={ta['max_coarse']} Q={Q} "
          f"K={K} stride={ta['temporal_stride']} | ckpt epoch={ck.get('epoch')} "
          f"val_total={ck.get('val_total')}")

    amp_dtype = args.amp_dtype or ta.get("amp_dtype", "bf16")
    amp_enabled = (amp_dtype == "bf16") and dev.type == "cuda"
    anytop_root = args.anytop_root or ta.get("anytop_root") or ck.get("data_root")

    # Resolve clip frame length: override the frozen ckpt's max_frames when
    # --num_frames is given (full-length backbone export uses 300). T_lat is the
    # latent frame count at the tokenizer's temporal_stride (ceil for the partial
    # tail window the conv stack keeps).
    num_frames = args.num_frames if args.num_frames is not None else ta.get("max_frames", 64)
    temporal_stride = ta["temporal_stride"]
    T_lat = math.ceil(num_frames / temporal_stride)
    print(f"Export frame length: num_frames={num_frames} "
          f"(ckpt max_frames={ta.get('max_frames', 64)}) -> T_lat={T_lat} "
          f"@ stride={temporal_stride}")

    # Caption caches ON (gap #3): VQ training used load_captions=False; the
    # CodeFlow backbone needs T5 mean + token captions, so the exporter must open
    # them. dual_text => return_caption_tokens=True (mean + token caches).
    ds_common = dict(
        num_frames=num_frames, max_joints=ta.get("max_joints", 64),
        val_frac=ta.get("val_frac", 0.05), seed=ta.get("seed", 42),
        data_root=anytop_root, load_captions=True,
        caption_emb_cache=args.caption_emb_cache,
        caption_token_cache=args.caption_token_cache,
        return_caption_tokens=True,
        caption_token_max_len=args.caption_token_max_len,
        random_caption=False,  # deterministic primary caption for the export
    )

    manifest = {"frozen_vqvae_ckpt": str(args.frozen_vqvae_ckpt),
                "ckpt_epoch": ck.get("epoch"), "D": D, "Q": Q, "K": K,
                "max_coarse": ta["max_coarse"], "temporal_stride": temporal_stride,
                "num_frames": num_frames, "T_lat": T_lat,
                "ckpt_max_frames": ta.get("max_frames", 64),
                "anytop_root": str(anytop_root), "amp_dtype": amp_dtype,
                "geo_inf_sentinel": GEO_INF_SENTINEL, "splits": {}}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ds = AnyTopDataset(split=split, **ds_common)
        n = len(ds) if args.max_clips <= 0 else min(args.max_clips, len(ds))
        # Disjoint 1/num_shards subset of clips by dataset index (==1 -> all clips).
        # The per-clip npz keeps its dataset-index filename {i:06d}.npz, so shards
        # write disjoint filenames into the SAME split dir with no collision.
        shard_indices = [i for i in range(n) if i % args.num_shards == args.shard_idx]
        if sharded:
            print(f"\n[{split}] {len(ds)} clips; shard {args.shard_idx}/{args.num_shards} "
                  f"-> {len(shard_indices)} clips")
        else:
            print(f"\n[{split}] {len(ds)} clips (exporting {n})")
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # ---- PREFLIGHT (fail-loud text-coverage gate) ----
        # Before any export, verify the clips actually carry text. A clip "has
        # text" iff caption_emb is not all-zero AND caption_token_mask.sum()>0.
        # Coverage below --min_text_coverage (or any required text field all-zero
        # across every clip) aborts loudly: never silently emit a text-less cache.
        n_with_text = 0
        any_emb_nonzero = False
        any_tokmask_nonzero = False
        for i in shard_indices:
            item = ds[i]
            emb_nonzero = bool(item["caption_emb"].abs().sum().item() > 0)
            tokmask_sum = int(item["caption_token_mask"].sum().item())
            any_emb_nonzero = any_emb_nonzero or emb_nonzero
            any_tokmask_nonzero = any_tokmask_nonzero or (tokmask_sum > 0)
            if emb_nonzero and tokmask_sum > 0:
                n_with_text += 1
        n_shard = len(shard_indices)
        coverage = (n_with_text / n_shard) if n_shard > 0 else 0.0
        print(f"[{split}] PREFLIGHT text-coverage={coverage:.4f} "
              f"({n_with_text}/{n_shard}) >= min {args.min_text_coverage}")
        if (not any_emb_nonzero) or (not any_tokmask_nonzero):
            raise RuntimeError(
                f"[PREFLIGHT FAIL] split={split}: required text field is ALL-ZERO "
                f"across every clip "
                f"(caption_emb nonzero={any_emb_nonzero}, "
                f"caption_token_mask nonzero={any_tokmask_nonzero}). "
                f"emb cache={args.caption_emb_cache} | "
                f"token cache={args.caption_token_cache}. "
                f"Refusing to export a text-less cache.")
        if coverage < args.min_text_coverage:
            raise RuntimeError(
                f"[PREFLIGHT FAIL] split={split}: text-coverage {coverage:.4f} "
                f"({n_with_text}/{n_shard}) < min {args.min_text_coverage}. "
                f"emb cache={args.caption_emb_cache} | "
                f"token cache={args.caption_token_cache}. "
                f"Refusing to silently export a text-less/under-captioned cache; "
                f"lower --min_text_coverage explicitly for a deliberate "
                f"unconditional run.")

        index_rows = []
        max_id_err_fp32 = 0.0
        with torch.no_grad():
            for i in shard_indices:
                item = ds[i]
                raw = anytop_collate_fn([item])
                raw = {k: v.to(dev) if torch.is_tensor(v) else v
                       for k, v in raw.items()}
                batch = GraphMotionBatch.from_collate_dict(raw)

                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        enc = model.encode(batch)
                        vq = model.quantizer(enc["h_lat"], enc["token_mask"],
                                             allow_collectives=False)
                else:
                    enc = model.encode(batch)
                    vq = model.quantizer(enc["h_lat"], enc["token_mask"],
                                         allow_collectives=False)

                z_q = vq["quantized"].float()[0]            # [T_lat,C,D]
                indices = vq["indices"][0]                  # [T_lat,C,Q] long
                token_mask = enc["token_mask"][0]           # [T_lat,C] bool
                coarse_mask = enc["coarse_mask"][0]         # [C]
                frame_mask_lat = enc["frame_mask_lat"][0]   # [T_lat]
                pooled_adj = enc["pooled_adjacency"].float()[0]   # [C,C]
                pooled_geo = enc["pooled_geodesic"].float()[0]    # [C,C]
                pooled_skel = enc["pooled_skeleton_embeddings"].float()[0]  # [C,D]
                assignment = enc["assignment"].float()[0]   # [J,C]
                s_j = enc["s_j"].float()[0]                  # [J,D] (decode needs it)

                # ---- HARD audits (M1 gates) ----
                Tlat, C = token_mask.shape
                assert z_q.shape == (Tlat, C, D), f"z_q shape {tuple(z_q.shape)}"
                assert indices.shape == (Tlat, C, Q), f"indices shape {tuple(indices.shape)}"
                # padded IDs exactly -1 (all stages) on padded tokens.
                pad = ~token_mask
                if pad.any():
                    assert (indices[pad] == -1).all(), "padded token has non(-1) id"
                # valid tokens: FULL Q => every stage id in [0,K-1] (eval, no dropout).
                val = token_mask
                if val.any():
                    vi = indices[val]                       # [n_valid, Q]
                    assert (vi >= 0).all(), "valid token has -1 id (dropout leaked?)"
                    assert (vi < K).all(), "valid id >= K"
                # ids_to_embeddings(indices) ~= z_q on valid (fp32 audit).
                z_from_ids = model.ids_to_embeddings(
                    indices.unsqueeze(0), token_mask.unsqueeze(0))[0]  # [T_lat,C,D]
                if val.any():
                    err = (z_from_ids[val] - z_q[val]).abs().max().item()
                    max_id_err_fp32 = max(max_id_err_fp32, err)

                # ---- save (fp16 for the big tensors; +inf geo -> sentinel) ----
                geo_save = pooled_geo.clone()
                geo_save[torch.isinf(geo_save)] = GEO_INF_SENTINEL
                stem = item["motion_id"]
                npz_path = split_dir / f"{i:06d}.npz"
                np.savez_compressed(
                    npz_path,
                    z_q=z_q.cpu().numpy().astype(np.float16),
                    indices=indices.cpu().numpy().astype(np.int16),
                    token_mask=token_mask.cpu().numpy(),
                    coarse_mask=coarse_mask.cpu().numpy(),
                    frame_mask_lat=frame_mask_lat.cpu().numpy(),
                    pooled_adjacency=pooled_adj.cpu().numpy().astype(np.float16),
                    pooled_geodesic=geo_save.cpu().numpy().astype(np.float16),
                    pooled_skeleton_embeddings=pooled_skel.cpu().numpy().astype(np.float16),
                    assignment=assignment.cpu().numpy().astype(np.float16),
                    s_j=s_j.cpu().numpy().astype(np.float16),
                    joint_mask=batch.joint_mask[0].cpu().numpy(),
                    rest_offsets=batch.rest_offsets[0].cpu().numpy().astype(np.float32),
                    anytop_mean=batch.anytop_mean[0].cpu().numpy().astype(np.float32),
                    anytop_std=batch.anytop_std[0].cpu().numpy().astype(np.float32),
                    parent_indices=np.asarray(item["parent_indices"], dtype=np.int64),
                    num_joints=np.int64(item["num_joints"]),
                    caption_emb=item["caption_emb"].numpy().astype(np.float16),
                    caption_token_emb=item["caption_token_emb"].numpy().astype(np.float16),
                    caption_token_mask=item["caption_token_mask"].numpy(),
                    has_text=np.bool_(item["has_text"]),
                )
                index_rows.append({"idx": i, "file": npz_path.name,
                                   "motion_id": stem,
                                   "object_type": item["object_type"],
                                   "text": item.get("caption", ""),
                                   "num_joints": int(item["num_joints"]),
                                   "n_valid_tokens": int(token_mask.sum().item())})
                if (i + 1) % 50 == 0 or i == n - 1:
                    print(f"  [{split}] {i + 1}/{n}  id_err_fp32(max)={max_id_err_fp32:.2e}")

        if max_id_err_fp32 > args.identity_tol:
            raise RuntimeError(
                f"[{split}] ids_to_embeddings(indices) vs z_q fp32 max err "
                f"{max_id_err_fp32:.3e} > tol {args.identity_tol} (RVQ identity broken)")
        if sharded:
            # Per-shard index file (disjoint name -> no race on the shared split dir);
            # the merge step concatenates these into index.jsonl + writes manifest.json.
            shard_index = split_dir / f"index_shard{args.shard_idx:03d}.jsonl"
            shard_index.write_text(
                "\n".join(json.dumps(r) for r in index_rows) + "\n")
            # Per-shard manifest sidecar: global static fields + THIS shard's
            # max_id_err_fp32. The merge step reads these to assemble the final
            # manifest.json (CPU-only, no ckpt/torch reload). Same :03d zero-pad as
            # the index sidecar so the merge can read both by f"...{s:03d}...".
            shard_manifest = {k: v for k, v in manifest.items() if k != "splits"}
            shard_manifest["max_id_err_fp32"] = max_id_err_fp32
            (split_dir / f"manifest_shard{args.shard_idx:03d}.json").write_text(
                json.dumps(shard_manifest, indent=2))
            print(f"[shard {args.shard_idx}/{args.num_shards}] split={split} "
                  f"wrote {len(index_rows)} npz")
        else:
            (split_dir / "index.jsonl").write_text(
                "\n".join(json.dumps(r) for r in index_rows) + "\n")
            manifest["splits"][split] = {"n": n, "max_id_err_fp32": max_id_err_fp32}
            print(f"  [{split}] DONE {n} clips -> {split_dir} "
                  f"(RVQ-identity fp32 max err {max_id_err_fp32:.2e})")

    if sharded:
        print(f"\n[shard {args.shard_idx}/{args.num_shards}] export complete -> {out_dir} "
              f"(no manifest.json in sharded mode; the merge step writes it)")
    else:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nExport complete -> {out_dir}\nmanifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
