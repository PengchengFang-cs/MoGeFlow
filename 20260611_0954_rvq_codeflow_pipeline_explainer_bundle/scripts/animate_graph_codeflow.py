#!/usr/bin/env python3
"""Text-to-motion visual QA for Graph-CodeFlow (level_a / graph_pscf) — handoff/20260609_graph_
codeflow_rvq_backbone_plan.md §8 + LOCKED recipe (single-gif T2M layout).

Pipeline (inference, frozen tokenizer + trained flow):
  target skeleton + prompt
    -> tokenizer.prepare_skeleton_only(batch, T_lat)   (motion-independent graph)
    -> GraphCodeFlow.sample(cond, token_mask, ...)      (ODE + CFG -> z_hat)
    -> tokenizer.nearest_residual_ids(z_hat)            (residual snap -> indices)
    -> tokenizer.decode_from_indices(indices, meta, batch)  (frozen decoder)
    -> anytop13 motion [1,T,J,13]
    -> de-norm + rot6d-FK recovery -> single-gif T2M (static skeleton + prompt +
       pred; NO GT column — T2M inference takes only skeleton + prompt).

Read-only QA. Renders both the snapped-decode motion and (optionally) the
continuous-decode motion for the continuous-vs-snapped comparison (the key gate).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.animate import fk_rest_pose  # noqa: E402
from scripts.animate_denoiser import make_t2m_large_gif  # noqa: E402
from src.data.anytop_dataset import (  # noqa: E402
    AnyTopDataset, collate_fn as anytop_collate_fn, _STD_FLOOR,
)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402
from src.models.vq_model import GraphVQTokenizer  # noqa: E402
from src.models.CodeFlow_Model import GraphCodeFlow  # noqa: E402


def load_frozen_tokenizer(ckpt_path: str, dev: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = ck["args"]
    model = GraphVQTokenizer(
        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
        n_graph_layers=ta["n_graph_layers"],
        n_enc_temporal_layers=ta["n_enc_temporal_layers"],
        n_pre_vq_layers=ta["n_pre_vq_layers"], n_post_vq_layers=ta["n_post_vq_layers"],
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
    model.eval(); model.requires_grad_(False)
    return model, ta


def load_flow(ckpt_path: str, code_dim: int, dev: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    flow = GraphCodeFlow(
        code_dim=a.get("code_dim", code_dim), n_heads=a.get("n_heads", 8),
        d_ff=a.get("d_ff", 2048), n_layers=a.get("n_layers", 5),
        d_text=768, text_token_dim=768, dropout=a.get("dropout", 0.1),
        # model_variant + graph_pscf arch args (old ckpts: level_a defaults so they
        # still rebuild; graph_pscf ckpts carry depth_double/depth_single/mlp_ratio).
        model_variant=a.get("model_variant", "level_a"),
        depth_double=a.get("depth_double", 6), depth_single=a.get("depth_single", 12),
        max_T_lat=a.get("max_T_lat", 75), mlp_ratio=a.get("mlp_ratio", 4.0),
    ).to(dev)
    flow.load_state_dict(ck["model_state_dict"], strict=True)
    flow.eval(); flow.requires_grad_(False)
    return flow, ck


def encode_ood_text(text: str, dev: torch.device):
    """T5-encode an arbitrary string into (global[768], tokens[64,768], mask[64]),
    matching the TRAINING caption cache convention exactly (t5-base, max_length=64;
    global = mask-mean of last_hidden_state, tokens = padded [64,768] + [64] bool).
    See precompute_t5_captions.py (global) + precompute_t5_caption_tokens.py (tokens)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import T5EncoderModel, T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained("t5-base")
    t5 = T5EncoderModel.from_pretrained("t5-base").to(dev).eval()
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True,
              max_length=64).to(dev)
    with torch.no_grad():
        hs = t5(input_ids=enc.input_ids,
                attention_mask=enc.attention_mask).last_hidden_state[0]   # [64,768]
    m = enc.attention_mask[0].bool()                                      # [64]
    g = (hs * m.unsqueeze(-1).float()).sum(0) / m.sum().clamp_min(1)      # [768]
    return g.float(), hs.float(), m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--frozen_vqvae_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--species", default="PZ_Grey_Seal_Female,PZ_Caracal_Male,"
                    "PZ_West_African_Lion_Male,PZ_Red_Kangaroo_Female")
    ap.add_argument("--n_per", type=int, default=1)
    ap.add_argument("--cfg_scale", type=float, default=4.0,
                    help="CFG scale (SWEEP starting point — not hardcoded 6.0)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--T_lat", type=int, default=None,
                    help="latent frames to generate; defaults to num_frames/stride")
    ap.add_argument("--num_frames", type=int, default=None,
                    help="override the dataset num_frames used for rendering; None = "
                         "use ckpt max_frames. Set 300 to QA full-length clips")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--anytop_root", type=str, default=None)
    ap.add_argument("--caption_emb_cache", type=str,
                    default="data/anytop_caption_t5_cleanL5_multi.npz")
    ap.add_argument("--caption_token_cache", type=str,
                    default="data/anytop_caption_t5_cleanL5_multi")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ood_text", type=str, default=None,
                    help="OOD/custom text: T5-encode this string and OVERRIDE the picked "
                         "clip's caption (global+tokens+mask, has_text=True). The red GT "
                         "panel is dropped (the clip's GT is for its ORIGINAL caption, not "
                         "this text). For text-generalization experiments.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    dev = torch.device(args.device)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, ta = load_frozen_tokenizer(args.frozen_vqvae_ckpt, dev)
    flow, fck = load_flow(args.flow_ckpt, ta["d_model"], dev)
    stride = ta["temporal_stride"]
    # Render frame length: override the ckpt's max_frames when --num_frames is set
    # (full-length QA uses 300). T_lat defaults to num_frames/stride.
    num_frames = args.num_frames if args.num_frames is not None else ta.get("max_frames", 64)
    # graph_pscf trains at FULL length (T_lat up to max_T_lat=75); rendering at the
    # tokenizer's 64-frame default would mismatch the trained regime -> require an
    # explicit --num_frames (e.g. 300) instead of silently defaulting to 64.
    if getattr(flow, "model_variant", "level_a") == "graph_pscf" and args.num_frames is None:
        raise SystemExit("[QA FAIL] graph_pscf QA needs an explicit --num_frames "
                         "(full-length, e.g. 300); refusing the 64-frame default.")
    T_lat = args.T_lat or (num_frames // stride)
    T_full = T_lat * stride
    print(f"tokenizer code_dim={ta['d_model']} Q={ta['num_quantizers']} stride={stride}; "
          f"flow epoch={fck.get('epoch')} val_flow={fck.get('val_flow')}; "
          f"T_lat={T_lat} T_full={T_full} cfg={args.cfg_scale} steps={args.steps}")

    anytop_root = args.anytop_root or ta.get("anytop_root")
    ds = AnyTopDataset(
        split=args.split, num_frames=num_frames, max_joints=ta.get("max_joints", 64),
        val_frac=ta.get("val_frac", 0.05), seed=ta.get("seed", 42),
        data_root=anytop_root, load_captions=True,
        caption_emb_cache=args.caption_emb_cache,
        caption_token_cache=args.caption_token_cache,
        return_caption_tokens=True, random_caption=False)

    # OOD/custom text: encode once (single string applied to all picked clips).
    ood_fields = None
    if args.ood_text:
        ood_fields = encode_ood_text(args.ood_text, dev)
        print(f"[OOD-TEXT] {args.ood_text!r} -> global{tuple(ood_fields[0].shape)} "
              f"tokens{tuple(ood_fields[1].shape)} valid_tok={int(ood_fields[2].sum())}; "
              f"GT panel DROPPED (clip GT is for original caption, not this text)")

    want = [s.strip() for s in args.species.split(",") if s.strip()]
    picked = {s: 0 for s in want}
    summary = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            sp = item["object_type"]
            if sp not in picked or picked[sp] >= args.n_per:
                continue
            raw = anytop_collate_fn([item])
            raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
            batch = GraphMotionBatch.from_collate_dict(raw)

            # OOD-text override: replace ALL THREE text fields (graph_pscf uses both
            # token cross-attn and global-add, both has_text-gated) + set has_text.
            if ood_fields is not None:
                g, tok_emb, tok_mask = ood_fields
                batch.caption_emb = g.unsqueeze(0)             # [1,768]
                batch.caption_token_emb = tok_emb.unsqueeze(0)  # [1,64,768]
                batch.caption_token_mask = tok_mask.unsqueeze(0)  # [1,64]
                batch.has_text = torch.tensor([True], device=dev)

            # Per-sample generation length = the GT clip's TRUE length (user
            # 2026-06-10: "gt是多少长度，你就应该生成多少长度"). --num_frames is
            # only the dataset container / upper bound, not the generated length.
            gt_T = min(num_frames, int(item["num_frames"]))
            T_lat_i = min(T_lat, max(1, (gt_T + stride - 1) // stride))

            # Motion-independent graph metadata for the target skeleton.
            meta = tokenizer.prepare_skeleton_only(batch, T_lat_i)
            cond = {
                "text_global": batch.caption_emb.float(),
                "text_tokens": batch.caption_token_emb.float(),
                "text_token_mask": batch.caption_token_mask,
                "has_text": batch.has_text,
                "pooled_adjacency": meta["pooled_adjacency"].float(),
                "pooled_geodesic": meta["pooled_geodesic"].float(),
                "pooled_skeleton_embeddings": meta["pooled_skeleton_embeddings"].float(),
                "coarse_mask": meta["coarse_mask"],
                "frame_mask_lat": meta["frame_mask_lat"],
            }
            B, C = meta["coarse_mask"].shape
            # ODE + CFG sample -> continuous z_hat -> residual snap -> decode.
            z_hat = flow.sample(cond, meta["token_mask"], T_lat_i, C,
                                steps=args.steps, cfg_scale=args.cfg_scale,
                                validate_inputs=True)
            proj = tokenizer.nearest_residual_ids(z_hat, meta["token_mask"])
            indices_hat = proj["indices_hat"]
            fake_batch = type("B", (), {"joint_mask": batch.joint_mask})()
            snap = tokenizer.decode_from_indices(indices_hat, meta, fake_batch)["pred_motion"]
            cont = tokenizer.decode(z_hat, meta, fake_batch)["pred_motion"]

            J = int(item["num_joints"])
            std = raw["anytop_std"][0, :J].cpu().numpy()
            mean = raw["anytop_mean"][0, :J].cpu().numpy()
            parents = [int(p) for p in item["parent_indices"][:J]]
            offsets = np.asarray(item["rest_offsets"])[:J]

            def to_world(pred_motion):
                # truncate T_lat_i*stride (>= gt_T by ceil) down to the GT length
                pn = pred_motion[0, :gt_T, :J, :].float().cpu().numpy()
                pr = pn * (std[None] + _STD_FLOOR) + mean[None]
                return recover_from_bvh_rot_np(pr, parents, offsets)  # [T,J,3]

            snap_world = to_world(snap)
            cont_world = to_world(cont)
            static_pose = fk_rest_pose(offsets, parents)
            prompt_text = args.ood_text if ood_fields is not None else (item.get("caption") or "")
            p_spd = float(np.linalg.norm(np.diff(snap_world, axis=0), axis=-1).mean())
            # GT source clip (denoiser-large convention): motion_features ch0:3,
            # rendered RIGHTMOST in red. Same length as generation by design.
            # Under OOD text the clip's GT is for the ORIGINAL caption (unrelated to the
            # injected prompt) → drop the GT panel for honesty; speed_ratio also N/A.
            gt_world = batch.motion_features[0, :gt_T, :J, :3].float().cpu().numpy()
            g_spd = float(np.linalg.norm(np.diff(gt_world, axis=0), axis=-1).mean())
            if gt_world.shape[0] != snap_world.shape[0]:  # fail loud, no silent misalign
                raise SystemExit(f"[QA FAIL] GT/pred length mismatch: "
                                 f"{gt_world.shape[0]} vs {snap_world.shape[0]}")
            gt_for_render = None if ood_fields is not None else gt_world

            k = picked[sp]
            # Large-figure T2M (user 2026-06-10): PIL renderer, GT stitched as the
            # RIGHTMOST red panel → input | PRED snapped | PRED continuous | GT.
            gif = out_dir / f"{sp}_clip{k}_t2m.gif"
            make_t2m_large_gif(
                snap_world, cont_world, static_pose, parents, prompt_text,
                str(gif), fps=args.fps, gt=gt_for_render,
                pred_labels=("PRED snapped decode", "PRED continuous decode"))
            ratio_str = ("N/A(OOD)" if ood_fields is not None
                         else f"{p_spd / max(g_spd, 1e-9):.3f}")
            line = (f"{sp} clip{k}: J={J} T={gt_T} prompt={prompt_text[:50]!r} "
                    f"proj_err={proj['projection_error'].item():.4f} "
                    f"snap_speed={p_spd:.4f} GT_speed={g_spd:.4f} "
                    f"speed_ratio={ratio_str} cont_vs_snap_maxabs="
                    f"{(cont - snap).abs().max().item():.4f} -> {gif.name}")
            print(line)
            summary.append(line)
            picked[sp] += 1
            if all(picked[s] >= args.n_per for s in want):
                break

    (out_dir / "t2m_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nDONE {sum(picked.values())} gifs -> {out_dir}")
    print("PER-SPECIES picked:", picked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
