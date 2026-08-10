#!/usr/bin/env python3
"""Train Graph-CodeFlow (level_a probe / graph_pscf formal backbone — rectified-flow over the FROZEN Graph-VQVAE
post-RVQ z_q) — handoff/20260609_graph_codeflow_rvq_backbone_plan.md +
handoff/20260609_0530_graph_codeflow_locked_recipe_and_state.md (LOCKED recipe).

SEPARATE from train_graph_vqvae.py / train_graph_vae.py / train_denoiser.py. Reads
the offline RVQ token cache (scripts/export_graph_vq_tokens.py), trains
GraphCodeFlow (flow-only loss), and logs the continuous-vs-snapped projection QA
(THE key gate). Mirrors train_graph_vqvae.py's DDP / bf16-autocast / resume /
logging patterns so the operational behavior is familiar.

LOCKED recipe defaults: batch 64, lr 1e-4, half_cosine, warmup 2000,
eta_min_ratio 0.01, wd 0.01, grad_clip 1.0, cond_drop_prob 0.1,
flow_loss_weight 1.0, terminal/clean 0.0, seed 42, empirical z_q norm,
terminal ID CE OFF.

Modes:
  --smoke        : a few train iters + 1 val pass (single proc OK).
  --mem_profile  : one fwd+bwd at --batch_size, report peak CUDA mem, exit (does
                   NOT launch the real run).
  (default)      : full training loop. DO NOT launch the real run from this task
                   (the frozen tokenizer is still training to ep300).

Usage (smoke, single GPU):
  python scripts/train_graph_codeflow.py --token_cache /tmp/cf_tokens \
    --frozen_vqvae_ckpt /tmp/vqvae_cur.pt --smoke --out /tmp/cf_smoke --overwrite
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.vq_model import GraphVQTokenizer
from src.models.CodeFlow_Model import GraphCodeFlow
from src.models.CodeFlow_Model.token_dataset import TokenCacheDataset, token_collate


def _ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1, True
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    return True, rank, local_rank, world_size, rank == 0


def load_frozen_tokenizer(ckpt_path: str, dev: torch.device) -> GraphVQTokenizer:
    """Rebuild + freeze the Graph-VQVAE tokenizer (for the snapped-decode QA +
    empirical-norm decode path). eval() + requires_grad_(False)."""
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
    model.eval()
    model.requires_grad_(False)
    return model, ta


def build_cond(b: dict, cond_drop_prob: float, training: bool,
               dtype: torch.dtype):
    """Assemble the GraphStructuredCodeFlow conditioning dict from a token batch.

    has_text starts from the dataset flag; during training we additionally CFG-
    drop (flip True->False) with cond_drop_prob (so the model learns the uncond
    branch). Float conditioning is cast to `dtype` (fp32 unless bf16 autocast
    wraps the forward — the model enforces dtype-match on the fp32 path).
    """
    has_text = b["has_text"].clone()
    if training and cond_drop_prob > 0.0:
        drop = torch.rand(has_text.shape[0], device=has_text.device) < cond_drop_prob
        has_text = has_text & ~drop
    return {
        "text_global": b["caption_emb"].to(dtype),
        "text_tokens": b["caption_token_emb"].to(dtype),
        "text_token_mask": b["caption_token_mask"],
        "has_text": has_text,
        "pooled_adjacency": b["pooled_adjacency"].to(dtype),
        "pooled_geodesic": b["pooled_geodesic"].to(dtype),
        "pooled_skeleton_embeddings": b["pooled_skeleton_embeddings"].to(dtype),
        "coarse_mask": b["coarse_mask"],
        "frame_mask_lat": b["frame_mask_lat"],
    }


def compute_empirical_stats(ds: TokenCacheDataset, D: int, dev: torch.device,
                            max_clips: int = 0, cache_path: str | None = None,
                            cache_key: dict | None = None,
                            is_ddp: bool = False, rank: int = 0):
    """Empirical z_q mean/std over VALID tokens of the train cache (LOCKED:
    empirical normalization over the FULL train set, not codebook-stat). Streamed
    sum / sumsq. max_clips<=0 (default) uses ALL train clips; a positive cap is for
    smoke/debug only (it would normalize on a PREFIX, not the full set).

    Startup acceleration (2026-06-09): (1) DISK CACHE — if cache_path exists and its
    stored cache_key matches, load mean/std instantly (skips the full scan; survives
    resume/restart/new experiments on the same cache). cache_key encodes the cache
    identity (manifest_md5 + full-content index_md5 + n/D/max_clips) so any re-export
    or clip-set/content change INVALIDATES it (a content hash, not a byte-size).
    (2) rank-0-only under DDP — only rank 0 scans (or loads) + writes the cache, then
    broadcasts mean/std to all ranks. Avoids every rank re-decompressing the full cache
    (the un-optimized path made a 6-rank/full-length scan take ~30min; this makes a cold
    scan ~6x faster and a warm/cached start instant)."""
    n = len(ds) if max_clips <= 0 else min(len(ds), max_clips)
    mean_t = std_t = None
    count = 0
    # Only rank 0 (or a single process) scans/loads + writes the cache.
    if not is_ddp or rank == 0:
        loaded = False
        if cache_path and Path(cache_path).exists():
            try:
                c = torch.load(cache_path, map_location="cpu", weights_only=False)
                if c.get("D") == D and c.get("cache_key") == cache_key:
                    mean_t, std_t = c["mean"].float(), c["std"].float()
                    count = int(c["count"])
                    # Explicit raise (not assert: survives `python -O`, which would
                    # strip an assert and silently feed a bad-shape mean/std). A bad
                    # shape here is caught by the enclosing except -> loaded=False ->
                    # full re-scan + cache rewrite (self-heals instead of using it).
                    if tuple(mean_t.shape) != (D,) or tuple(std_t.shape) != (D,):
                        raise RuntimeError(
                            f"cached empirical stats shape mismatch: "
                            f"mean{tuple(mean_t.shape)} std{tuple(std_t.shape)} != ({D},)")
                    loaded = True
            except Exception:
                loaded = False
        if not loaded:
            count = 0
            s = torch.zeros(D, dtype=torch.float64)
            s2 = torch.zeros(D, dtype=torch.float64)
            for i in range(n):
                it = ds[i]
                z = it["z_q"].reshape(-1, D).double()       # [T_lat*C, D]
                m = it["token_mask"].reshape(-1)
                zv = z[m]
                count += zv.shape[0]
                s += zv.sum(dim=0)
                s2 += zv.pow(2).sum(dim=0)
            if count == 0:
                raise RuntimeError("compute_empirical_stats: zero valid tokens in cache")
            mean_t = (s / count).float()
            std_t = ((s2 / count) - (s / count).pow(2)).clamp_min(1e-12).sqrt().float()
            if cache_path:
                try:
                    torch.save({"mean": mean_t, "std": std_t, "count": count,
                                "D": D, "cache_key": cache_key}, cache_path)
                except Exception:
                    pass
        mean_t, std_t = mean_t.to(dev), std_t.to(dev)
    else:
        mean_t = torch.zeros(D, dtype=torch.float32, device=dev)
        std_t = torch.ones(D, dtype=torch.float32, device=dev)
    # Broadcast rank-0's stats to all ranks (DDP); every rank ends with identical norm.
    if is_ddp:
        dist.broadcast(mean_t, src=0)
        dist.broadcast(std_t, src=0)
        cnt = torch.tensor([count], dtype=torch.long, device=dev)
        dist.broadcast(cnt, src=0)
        count = int(cnt.item())
    return mean_t, std_t, count


@torch.no_grad()
def projection_qa(flow, tokenizer, b: dict, cond: dict, dev: torch.device,
                  decode: bool = False):
    """THE key gate: compare continuous decode(z_hat) vs snapped decode(z_snap)
    and report projection_error = mse(z_hat, z_snap) over valid tokens.

    Here z_hat is the model's predicted CLEAN latent from a single flow eval at a
    fixed t (denormalized to raw RVQ space), NOT a full ODE sample (cheap, per-step
    diagnostic). Returns projection_error, per-q generated-code usage, and (if
    decode=True) the max abs decoded-motion gap continuous-vs-snapped.
    """
    z_q = b["z_q"].to(dev)
    token_mask = b["token_mask"].to(dev)
    B, T_lat, C, D = z_q.shape
    # One flow eval at t~U: predict v -> clean (in normalized space) -> denorm.
    x = flow.normalize(z_q) * token_mask.unsqueeze(-1).float()
    noise = torch.randn_like(x) * flow.noise_scale * token_mask.unsqueeze(-1).float()
    t = torch.rand(B, device=dev)
    t_view = t[:, None, None, None]
    z_t = (t_view * x + (1.0 - t_view) * noise) * token_mask.unsqueeze(-1).float()
    v = flow.predict_velocity(z_t, t, cond)
    clean = flow.predict_clean_from_velocity(z_t, t, v)
    z_hat = flow.denormalize(clean) * token_mask.unsqueeze(-1).float()

    proj = tokenizer.nearest_residual_ids(z_hat, token_mask)
    indices_hat = proj["indices_hat"]
    # per-q generated code usage (#unique codes used on valid tokens).
    usage = []
    for qi in range(indices_hat.shape[-1]):
        ids_q = indices_hat[..., qi][token_mask]
        usage.append(int(torch.unique(ids_q[ids_q >= 0]).numel()))
    out = {"projection_error": float(proj["projection_error"].item()),
           "code_usage_per_q": usage}
    if decode:
        skel_meta = {
            "s_j": b["s_j"].to(dev), "assignment": b["assignment"].to(dev),
            "coarse_mask": b["coarse_mask"].to(dev),
            "frame_mask_lat": b["frame_mask_lat"].to(dev),
            "pooled_adjacency": b["pooled_adjacency"].to(dev),
            "pooled_geodesic": b["pooled_geodesic"].to(dev),
        }
        fake_batch = SimpleNamespace(joint_mask=b["joint_mask"].to(dev))
        cont = tokenizer.decode(z_hat, skel_meta, fake_batch)["pred_motion"]
        snap = tokenizer.decode_from_indices(indices_hat, skel_meta, fake_batch)["pred_motion"]
        out["decode_cont_finite"] = bool(torch.isfinite(cont).all())
        out["decode_snap_finite"] = bool(torch.isfinite(snap).all())
        out["decode_cont_vs_snap_maxabs"] = float((cont - snap).abs().max().item())
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    # data / tokenizer
    p.add_argument("--token_cache", required=True, help="dir from export_graph_vq_tokens.py")
    p.add_argument("--frozen_vqvae_ckpt", required=True)
    # model
    p.add_argument("--model_variant", choices=["level_a", "graph_pscf"],
                   default="graph_pscf",
                   help="level_a = GraphStructuredCodeFlow probe (compat/smoke); "
                        "graph_pscf = 287M formal backbone (DEFAULT for formal "
                        "training, spec §5.3)")
    p.add_argument("--code_dim", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=5,
                   help="level_a only (graph_pscf uses depth_double/depth_single)")
    p.add_argument("--depth_double", type=int, default=6,
                   help="graph_pscf double-stream depth (spec §5.3)")
    p.add_argument("--depth_single", type=int, default=12,
                   help="graph_pscf single-stream depth (spec §5.3)")
    p.add_argument("--hidden_size", type=int, default=512,
                   help="graph_pscf hidden size; spec A3 pins H==D==code_dim==512 "
                        "for v1 (asserted below)")
    p.add_argument("--mlp_ratio", type=float, default=4.0,
                   help="graph_pscf DiT MLP expansion ratio (spec §5.3)")
    p.add_argument("--max_T_lat", type=int, default=75,
                   help="frame_seed / positional capacity = ceil(T_fine_max/temporal_stride); "
                        "full-length T_fine_max=300 stride 4 -> 75. Must be >= the token "
                        "cache's T_lat")
    p.add_argument("--dropout", type=float, default=None,
                   help="None = per-variant default (graph_pscf 0.05, level_a 0.1); "
                        "an explicit value always wins")
    # train (LOCKED recipe)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr_scheduler", choices=["half_cosine", "none"], default="half_cosine")
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--eta_min_ratio", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cond_drop_prob", type=float, default=0.1)
    p.add_argument("--flow_loss_weight", type=float, default=1.0)
    p.add_argument("--terminal_loss_weight", type=float, default=0.0)
    p.add_argument("--clean_loss_weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--empirical_stats_max_clips", type=int, default=0,
                   help="0 (default) = use ALL train clips for the empirical z_q "
                        "norm (LOCKED: full train-set stats). A positive value caps "
                        "to a PREFIX — smoke/debug only, NOT the real run.")
    # eval / cfg
    p.add_argument("--eval_cond_scale", type=float, default=4.0,
                   help="CFG scale for sampling QA — SWEEP starting point, NOT a "
                        "fixed default (project energy-overshoot history; recipe "
                        "says do not hardcode 6.0).")
    p.add_argument("--eval_steps", type=int, default=50)
    # logging / ckpt
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--qa_every", type=int, default=200,
                   help="run the decode-based continuous-vs-snapped QA every N steps")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_iters", type=int, default=4)
    p.add_argument("--mem_profile", action="store_true",
                   help="one fwd+bwd at --batch_size, report peak CUDA mem, exit")
    args = p.parse_args()

    is_ddp, rank, local_rank, world_size, is_main = _ddp_setup()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if is_ddp:
        dev = torch.device("cuda", local_rank)
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("[DEVICE FAIL] --device cuda but CUDA unavailable.")
        dev = torch.device(args.device)

    out_dir = Path(args.out)
    resume_in_place = (args.resume is not None
                       and out_dir.resolve() == Path(args.resume).resolve().parent)
    if (out_dir.exists() and any(out_dir.iterdir())
            and not args.overwrite and not resume_in_place):
        raise RuntimeError(f"[OUT FAIL] {out_dir} non-empty. Use --overwrite or "
                           f"--resume to continue in place.")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.jsonl"

    def log(msg: str) -> None:
        if not is_main:
            return
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    import subprocess
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git_sha = "unknown"
    log(f"=== Graph-CodeFlow [{args.model_variant}] rectified flow over post-RVQ z_q ===")
    log(f"git_sha: {git_sha}  device: {dev}  world_size: {world_size}")
    log(f"args: {vars(args)}")

    # ---- Frozen tokenizer (for snapped-decode QA + projection) ----
    tokenizer, ta = load_frozen_tokenizer(args.frozen_vqvae_ckpt, dev)
    D = ta["d_model"]
    if D != args.code_dim:
        raise RuntimeError(f"[CFG FAIL] tokenizer code_dim {D} != --code_dim {args.code_dim}")
    log(f"frozen tokenizer: code_dim={D} Q={ta['num_quantizers']} K={ta['num_codes']} "
        f"max_coarse={ta['max_coarse']}")

    # ---- Data (offline token cache) ----
    ds_train = TokenCacheDataset(args.token_cache, "train")
    try:
        ds_val = TokenCacheDataset(args.token_cache, "val")
    except FileNotFoundError:
        ds_val = None
    log(f"token cache: train={len(ds_train)}" + (f" val={len(ds_val)}" if ds_val else " (no val)"))
    if len(ds_train) < args.batch_size and not (args.smoke or args.mem_profile):
        raise RuntimeError(f"[DATA FAIL] train {len(ds_train)} < batch {args.batch_size}")
    # Preflight (P2): graph_pscf's frame_seed capacity is max_T_lat; the cache's T_lat
    # must not exceed it, else GraphPSCFFlowNet.forward raises mid-run. Catch it HERE
    # (right after cache load, before empirical stats / first forward) by reading the
    # first clip's z_q[T_lat,C,D].
    if args.model_variant == "graph_pscf" and len(ds_train) > 0:
        _cache_T_lat = int(ds_train[0]["z_q"].shape[0])
        if _cache_T_lat > args.max_T_lat:
            raise RuntimeError(
                f"[CFG FAIL] token cache T_lat={_cache_T_lat} > --max_T_lat={args.max_T_lat}: "
                f"graph_pscf frame_seed capacity too small. Re-export with smaller "
                f"num_frames, or raise --max_T_lat to >= {_cache_T_lat}.")
        log(f"preflight: cache T_lat={_cache_T_lat} <= max_T_lat={args.max_T_lat} OK")

    # ---- Resume: rebuild the model from the CKPT's saved architecture args ----
    # The model is constructed BEFORE the resume state_dict load below, so when
    # resuming we must build the SAME architecture the ckpt was trained with (its
    # model_variant + depths), not the CLI default — otherwise load_state_dict
    # (strict=True) would mismatch. Old ckpts (pre-model_variant) default to
    # level_a so they still load. Only architecture-defining args are overridden;
    # train/schedule args stay from the CLI.
    if args.resume is not None and Path(args.resume).exists():
        _rargs = torch.load(args.resume, map_location="cpu",
                            weights_only=False).get("args", {})
        for _k, _default in (("model_variant", "level_a"), ("code_dim", None),
                             ("n_heads", None), ("d_ff", None), ("n_layers", None),
                             ("depth_double", 6), ("depth_single", 12),
                             ("hidden_size", None), ("mlp_ratio", 4.0),
                             ("dropout", None), ("max_T_lat", 75)):
            _v = _rargs.get(_k, _default) if isinstance(_rargs, dict) \
                else getattr(_rargs, _k, _default)
            if _v is not None:
                setattr(args, _k, _v)
        log(f"resume: rebuilding model from ckpt args model_variant={args.model_variant} "
            f"depth_double={args.depth_double} depth_single={args.depth_single}")

    # Resolve dropout per variant (spec §5.4: graph_pscf=0.05, level_a=0.1). None =
    # not set on the CLI; an explicit --dropout wins, and resume above may have restored
    # the ckpt's dropout (non-None -> takes precedence). Resolving HERE — before
    # vars(args) is saved into the ckpt — records the real dropout so resume rebuilds it.
    if args.dropout is None:
        args.dropout = 0.05 if args.model_variant == "graph_pscf" else 0.1

    # ---- Model ----
    # spec A3: graph_pscf pins H==D==512 for v1 (no "H!=D is fine" speculation).
    if args.hidden_size != args.code_dim:
        raise RuntimeError(
            f"[CFG FAIL] spec A3 requires hidden_size==code_dim (H==D==512 for v1), "
            f"got hidden_size={args.hidden_size} code_dim={args.code_dim}")
    flow = GraphCodeFlow(
        code_dim=args.code_dim, n_heads=args.n_heads, d_ff=args.d_ff,
        n_layers=args.n_layers, d_text=768, text_token_dim=768, dropout=args.dropout,
        model_variant=args.model_variant, depth_double=args.depth_double,
        depth_single=args.depth_single, mlp_ratio=args.mlp_ratio,
        max_T_lat=args.max_T_lat,
    ).to(dev)
    # Empirical z_q normalization (LOCKED): mean/std over valid train tokens.
    # Startup acceleration: disk-cache the stats (keyed by cache identity from the
    # manifest, so a re-export invalidates it) + rank-0 scan + broadcast under DDP.
    _cache_path = None
    _cache_key = None
    try:
        import hashlib
        _tc = Path(args.token_cache)
        _man_text = (_tc / "manifest.json").read_text()
        _idx_bytes = (_tc / "train" / "index.jsonl").read_bytes()
        # Strong invalidation: full-manifest md5 (any export-config change) + a full
        # CONTENT md5 of train/index.jsonl (any clip-set/order/content change, even at
        # identical byte-size) + n/D/max_clips. A re-export rewrites manifest/index ->
        # key changes -> cache auto-invalidates. (md5 of ~24MB index is one-time
        # rank-0-only ~tens of ms, negligible vs the scan it guards.)
        _cache_key = {"manifest_md5": hashlib.md5(_man_text.encode()).hexdigest(),
                      "index_md5": hashlib.md5(_idx_bytes).hexdigest(),
                      "n": len(ds_train), "D": D,
                      "max_clips": args.empirical_stats_max_clips}
        _cache_path = str(_tc / "empirical_stats.pt")
    except Exception:
        _cache_path = None  # no manifest/index -> skip cache, fall back to full scan
    e_mean, e_std, n_stat = compute_empirical_stats(
        ds_train, D, dev, max_clips=args.empirical_stats_max_clips,
        cache_path=_cache_path, cache_key=_cache_key, is_ddp=is_ddp, rank=rank)
    flow.set_latent_stats(e_mean, e_std)
    log(f"empirical z_q norm over {n_stat} valid tokens: "
        f"mean|.|avg={e_mean.abs().mean().item():.4f} std.avg={e_std.mean().item():.4f}")
    n_params = sum(pp.numel() for pp in flow.parameters() if pp.requires_grad)
    log(f"GraphCodeFlow trainable params: {n_params:,}")

    amp_enabled = (args.amp_dtype == "bf16") and dev.type == "cuda"
    fwd_dtype = torch.float32  # conditioning/z_q dtype the model validates on the fp32 path
    amp_ctx = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
               if amp_enabled else contextlib.nullcontext)
    log(f"AMP: amp_dtype={args.amp_dtype} (autocast around fp32 flow math)")

    # ---- Mem profile (one fwd+bwd, report peak, exit; NOT a real launch) ----
    if args.mem_profile:
        bs = min(args.batch_size, len(ds_train))
        dl = DataLoader(ds_train, batch_size=bs, shuffle=True,
                        collate_fn=token_collate, num_workers=0, drop_last=True)
        b = next(iter(dl))
        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
        cond = build_cond(b, args.cond_drop_prob, training=True, dtype=fwd_dtype)
        torch.cuda.reset_peak_memory_stats(dev)
        with amp_ctx():
            r = flow.flow_loss(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
                               validate_inputs=True)
        loss = args.flow_loss_weight * r["flow_loss"]
        loss.backward()
        peak = torch.cuda.max_memory_allocated(dev) / 1e9
        log(f"[MEM PROFILE] batch_size={bs} flow_loss={r['flow_loss'].item():.4f} "
            f"peak_cuda_mem={peak:.2f} GB  (NO real run launched)")
        if is_ddp:
            dist.destroy_process_group()
        return 0

    train_sampler = (DistributedSampler(ds_train, shuffle=True, drop_last=True)
                     if is_ddp else None)
    nw = max(0, args.num_workers)
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=token_collate, num_workers=nw,
        drop_last=True, pin_memory=True, persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None))
    dl_val = (DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                         collate_fn=token_collate, num_workers=max(1, nw // 2),
                         drop_last=False, pin_memory=True)
              if ds_val is not None else None)

    if is_ddp:
        flow = DDP(flow, device_ids=[local_rank], find_unused_parameters=False)
    raw_flow = flow.module if is_ddp else flow

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(dl_train))
    total_steps = max(1, args.epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        """Linear warmup -> half-cosine decay to eta_min_ratio*lr (CodeFlow recipe)."""
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        if args.lr_scheduler == "none":
            return args.lr
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return args.lr * (args.eta_min_ratio + (1.0 - args.eta_min_ratio) * cos)

    n_iter, start_epoch, best_val = 0, 0, float("inf")
    if args.resume is not None:
        if not Path(args.resume).exists():
            raise RuntimeError(f"[RESUME FAIL] {args.resume} missing.")
        rc = torch.load(args.resume, map_location=dev, weights_only=False)
        raw_flow.load_state_dict(rc["model_state_dict"], strict=True)
        opt.load_state_dict(rc["optimizer_state_dict"])
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(dev)
        start_epoch = int(rc["epoch"]) + 1
        n_iter = int(rc["global_step"])
        best_val = float(rc.get("best_val", float("inf")))
        log(f"resumed: start_epoch={start_epoch} n_iter={n_iter} best_val={best_val:.4f}")
        del rc

    n_epochs = 2 if args.smoke else args.epochs
    smoke_cap = args.smoke_iters if args.smoke else None

    for epoch in range(start_epoch, n_epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        flow.train()
        t0 = time.time()
        run_sum, run_cnt = 0.0, 0
        for it, b in enumerate(dl_train):
            if smoke_cap is not None and it >= smoke_cap:
                break
            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
            cond = build_cond(b, args.cond_drop_prob, training=True, dtype=fwd_dtype)
            with amp_ctx():
                r = flow(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
                         validate_inputs=(it == 0 and epoch == start_epoch))
            loss = args.flow_loss_weight * r["flow_loss"]

            if it == 0 and epoch == start_epoch:
                B, T_lat, C, Dd = b["z_q"].shape
                Qd = b["indices"].shape[-1]
                log(f"  [gate ok] z_q=[{B},{T_lat},{C},{Dd}] indices Q={Qd} "
                    f"flow_loss={r['flow_loss'].item():.4f}")
            if not torch.isfinite(loss):
                log(f"[GATE FAIL] loss non-finite at iter {n_iter}")
                return 1
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                log(f"[GATE FAIL] non-finite grad norm at iter {n_iter}")
                return 1
            cur_lr = lr_at(n_iter)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            opt.step()

            run_sum += float(r["flow_loss"].detach()); run_cnt += 1
            do_log = (n_iter % args.log_every == 0) or (it == 0 and epoch == start_epoch)
            do_qa = (n_iter % args.qa_every == 0) or (args.smoke and it == 0)
            if do_log or do_qa:
                qa = None
                if do_qa and is_main:
                    qa = projection_qa(raw_flow, tokenizer, b,
                                       build_cond(b, 0.0, training=False, dtype=fwd_dtype),
                                       dev, decode=do_qa)
                if do_log:
                    log(f"[ep{epoch} it{it} n_iter={n_iter}] flow_loss={r['flow_loss'].item():.5f} "
                        f"grad_norm={grad_norm.item():.3f} lr={cur_lr:.3e}"
                        + (f" | proj_err={qa['projection_error']:.4f} "
                           f"code_usage/q={qa['code_usage_per_q']}" if qa else ""))
                    if qa and "decode_cont_vs_snap_maxabs" in qa:
                        log(f"           [QA decode] cont_finite={qa['decode_cont_finite']} "
                            f"snap_finite={qa['decode_snap_finite']} "
                            f"cont_vs_snap_maxabs={qa['decode_cont_vs_snap_maxabs']:.4f}")
                    if is_main:
                        row = {"epoch": epoch, "iter": it, "n_iter": n_iter,
                               "flow_loss": r["flow_loss"].item(),
                               "grad_norm": grad_norm.item(), "lr": cur_lr}
                        if qa:
                            row.update(qa)
                        with open(metrics_path, "a") as f:
                            f.write(json.dumps(row) + "\n")
            n_iter += 1

        train_flow = run_sum / max(1, run_cnt)
        log(f"=== epoch {epoch} done in {time.time() - t0:.1f}s | train_flow={train_flow:.5f} ===")

        do_val = ((epoch + 1) % args.save_every == 0 or epoch == n_epochs - 1 or args.smoke)
        if do_val and is_main and dl_val is not None:
            raw_flow.eval()
            vlosses, vproj = [], []
            with torch.no_grad():
                for vit, vb in enumerate(dl_val):
                    if args.smoke and vit >= 2:
                        break
                    vb = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in vb.items()}
                    vcond = build_cond(vb, 0.0, training=False, dtype=fwd_dtype)
                    with amp_ctx():
                        vr = raw_flow.flow_loss(vb["z_q"].to(fwd_dtype), vb["token_mask"], vcond)
                    vlosses.append(vr["flow_loss"].item())
                    vproj.append(projection_qa(raw_flow, tokenizer, vb, vcond, dev,
                                               decode=False)["projection_error"])
            val_flow = float(np.mean(vlosses)) if vlosses else float("nan")
            val_proj = float(np.mean(vproj)) if vproj else float("nan")
            log(f"  [val] flow_loss={val_flow:.5f} projection_error={val_proj:.4f}")
            if not args.smoke:
                hist_best = min(best_val, val_flow)
                ckpt = {"model_state_dict": raw_flow.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "epoch": epoch, "global_step": n_iter, "args": vars(args),
                        "val_flow": val_flow, "val_proj": val_proj, "best_val": hist_best,
                        "git_sha": git_sha, "frozen_vqvae_ckpt": args.frozen_vqvae_ckpt,
                        "latent_mean": raw_flow.latent_mean.cpu(),
                        "latent_std": raw_flow.latent_std.cpu()}
                torch.save(ckpt, out_dir / "last_model.pt")
                if val_flow < best_val:
                    best_val = val_flow
                    torch.save(ckpt, out_dir / "best_model.pt")
                    log(f"  [ckpt] new best val_flow={val_flow:.5f}")
            raw_flow.train()
        if is_ddp:
            dist.barrier()

    log("=== training loop complete ===")
    if is_ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
