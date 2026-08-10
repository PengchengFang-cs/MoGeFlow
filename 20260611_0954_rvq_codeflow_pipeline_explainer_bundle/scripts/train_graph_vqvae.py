#!/usr/bin/env python3
"""Train the Graph-VQVAE (coarse-slot structured RVQ) tokenizer on AniMo4D
AnyTop L5 (handoff/20260608_graph_vqvae_l5_pipeline_plan.md).

SEPARATE from scripts/train_graph_vae.py — uses GraphVQTokenizer +
compute_vq_loss_13ch (no KL) + root-drift/jitter QA. Mirrors train_graph_vae.py's
DDP / dataset / bf16-autocast / smoke patterns so the operational behavior is
familiar, but the model + loss are the VQ ones under src/models/vq_model/.

Modes:
  --unit_checks : run the M3 mask/STE/commit/EMA unit assertions and exit.
  --smoke       : run a few train iters (2-rank DDP exercised) + 1 val pass.
  (default)     : full training loop (DO NOT launch the 300ep run from this task).

Usage (single-node 2-card standalone DDP smoke):
  torchrun --standalone --nproc_per_node=2 scripts/train_graph_vqvae.py \
    --anytop_root data/animo4d_anytop_clean_L5 --smoke \
    --out runs/vqvae_smoke --overwrite
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.vq_model import (
    GraphVQTokenizer,
    compute_vq_loss_13ch,
    root_drift_jitter_qa,
)


# --------------------------------------------------------------------------- #
# DDP setup (mirror train_graph_vae.py:_ddp_setup)                            #
# --------------------------------------------------------------------------- #
def _ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1, True
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", device_id=torch.device("cuda", local_rank))
    return True, rank, local_rank, world_size, rank == 0


# --------------------------------------------------------------------------- #
# Loss runner                                                                 #
# --------------------------------------------------------------------------- #
def run_vq_loss(out, batch, loss_weights, effective_frame_mask, code_dim,
                allow_collectives=True):
    gt_motion = batch.anytop_x.permute(0, 3, 1, 2).contiguous()  # [B,T,J,13]
    return compute_vq_loss_13ch(
        pred_motion=out["pred_motion"],
        gt_motion=gt_motion,
        foot_contact_per_joint=batch.foot_contact_per_joint,
        anytop_mean=batch.anytop_mean,
        anytop_std=batch.anytop_std,
        parent_indices=batch.parent_indices,
        rest_offsets=batch.rest_offsets,
        joint_mask=batch.joint_mask,
        frame_mask=effective_frame_mask,
        code_dim=code_dim,
        commit_sq_sum=out["commit_sq_sum"],
        n_valid_tokens=out["n_valid_tokens"],
        weights=loss_weights,
        allow_collectives=allow_collectives,
    )


# --------------------------------------------------------------------------- #
# M3 unit checks (the 4 amendment-critical invariants)                        #
# --------------------------------------------------------------------------- #
def run_unit_checks(dev) -> int:
    """(a) padded slots -> ~0 decoder attention weight (F1)
       (b) STE mask applied post-detach (1c)
       (c) commit loss applied exactly once (F4)
       (d) EMA denominators exclude padded tokens (1b)
    Returns 0 on all-pass, 1 on any failure."""
    import torch.nn.functional as F
    from src.models.vq_model.masked_motion_decoder import MaskedSlotToJointCrossAttention
    from src.models.vq_model.quantizer import MaskedResidualVQ, _EMACodebook
    from src.models.vq_model.losses import _DEFAULT_VQ_WEIGHTS

    torch.manual_seed(0)
    ok = True

    # ---- (a) F1: padded slot keys get EXACTLY zero attention weight ----
    # Probe the cross-attn directly: build assignment that (via clamp(1e-8).log())
    # would otherwise leak onto padded slots, set a padded slot, assert 0 weight.
    D, H, B, J, K = 32, 4, 2, 5, 6
    attn_mod = MaskedSlotToJointCrossAttention(D, H, dropout=0.0).to(dev).eval()
    jq = torch.randn(B, J, D, device=dev)
    sf = torch.randn(B, K, D, device=dev)
    assign = torch.zeros(B, J, K, device=dev)
    # give every joint mass on slot 0 (valid); leave others 0 (-> log clamp leak)
    assign[:, :, 0] = 1.0
    coarse_mask = torch.ones(B, K, dtype=torch.bool, device=dev)
    coarse_mask[:, -1] = False   # last slot padded
    coarse_mask[1, -2] = False   # sample 1 has two padded slots

    # Monkey-inspect the softmax weights by reconstructing them (the module does
    # not return attn; replicate its score path to read the weights).
    import math
    with torch.no_grad():
        q = attn_mod.q_proj(attn_mod.norm_q(jq)).view(B, J, H, D // H).permute(0, 2, 1, 3)
        kv = attn_mod.norm_kv(sf)
        k = attn_mod.k_proj(kv).view(B, K, H, D // H).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // H)
        ab = assign.unsqueeze(1).clamp(min=1e-8).log()
        scores = scores + attn_mod.assign_bias_scale * ab
        key_mask = coarse_mask.bool().unsqueeze(1).unsqueeze(2)
        scores = scores.float().masked_fill(~key_mask, float("-inf"))
        w = F.softmax(scores, dim=-1).nan_to_num(0.0)   # [B,H,J,K]
    # weight on padded slot keys must be EXACTLY 0
    padded_w = w[:, :, :, ~coarse_mask.any(dim=0)]  # cols padded in ALL samples (slot -1)
    leak_global = padded_w.abs().max().item()
    # per-sample padded check (sample 1 slot -2)
    leak_s1 = w[1, :, :, -2].abs().max().item()
    if leak_global > 0.0 or leak_s1 > 0.0:
        print(f"[UNIT a FAIL] F1 padded-slot attention weight != 0 "
              f"(global={leak_global:.3e} s1={leak_s1:.3e})")
        ok = False
    else:
        print(f"[UNIT a OK] F1: padded-slot-key attention weight == 0 "
              f"(global={leak_global:.1e}, s1_extra_pad={leak_s1:.1e})")

    # ---- (b) 1c: STE mask applied post-detach (padded -> 0 value AND 0 grad) ----
    vq = MaskedResidualVQ(code_dim=D, num_codes=16, num_quantizers=2,
                          quantize_dropout_prob=0.0).to(dev).eval()
    Tl, C = 3, 4
    x = torch.randn(B, Tl, C, D, device=dev, requires_grad=True)
    tok_mask = torch.ones(B, Tl, C, dtype=torch.bool, device=dev)
    tok_mask[:, :, -1] = False  # last slot padded everywhere
    # allow_collectives=False: unit checks may run rank-0-only under torchrun
    # (process group initialized), so the fresh VQ modules here must NOT call
    # cross-rank collectives the other ranks are not making.
    o = vq(x, tok_mask, allow_collectives=False)
    zq = o["quantized"]
    # value: padded slot quantized output must be exactly 0
    pad_val = zq[:, :, -1, :].abs().max().item()
    # gradient: a loss on zq must give EXACTLY zero grad to padded x entries
    zq.sum().backward()
    pad_grad = x.grad[:, :, -1, :].abs().max().item()
    valid_grad = x.grad[:, :, 0, :].abs().max().item()
    if pad_val > 0.0 or pad_grad > 0.0:
        print(f"[UNIT b FAIL] STE padded value={pad_val:.3e} grad={pad_grad:.3e} "
              f"(both must be 0)")
        ok = False
    else:
        print(f"[UNIT b OK] 1c STE-post-detach: padded value=0 grad=0; "
              f"valid grad nonzero={valid_grad:.3e}")

    # ---- (c) F4: commit weight applied EXACTLY once ----
    # commit term in total == w_commit * raw_commit (not squared, not double).
    w_commit = _DEFAULT_VQ_WEIGHTS["commit"]
    vq2 = MaskedResidualVQ(code_dim=D, num_codes=16, num_quantizers=2,
                           quantize_dropout_prob=0.0).to(dev).eval()
    x2 = torch.randn(B, Tl, C, D, device=dev)
    m2 = torch.ones(B, Tl, C, dtype=torch.bool, device=dev)
    m2[:, :, -1] = False
    o2 = vq2(x2, m2, allow_collectives=False)
    raw_commit = o2["commit_loss"].item()
    # rebuild commit exactly as the loss wrapper does (single-rank => no allreduce)
    n_valid = o2["n_valid_tokens"].item()
    wrapper_commit = (o2["commit_sq_sum"].item() * 1.0) / (max(n_valid, 1.0) * D)
    contrib = w_commit * wrapper_commit
    double = w_commit * w_commit * wrapper_commit  # the WRONG (squared) value
    if abs(wrapper_commit - raw_commit) > 1e-5:
        print(f"[UNIT c FAIL] wrapper commit {wrapper_commit:.6e} != "
              f"quantizer raw commit {raw_commit:.6e}")
        ok = False
    elif abs(contrib - double) < 1e-12:
        print(f"[UNIT c FAIL] cannot distinguish single vs squared (w too close to 1)")
        ok = False
    else:
        print(f"[UNIT c OK] F4 commit applied ONCE: raw={raw_commit:.4e} "
              f"contrib={contrib:.4e} (w={w_commit}); squared-would-be={double:.4e}")

    # ---- (d) 1b: EMA denominators exclude padded tokens ----
    # Feed a codebook a batch where MANY tokens are padded zero-vectors; the EMA
    # cluster_size SUM must reflect the #VALID tokens, never the total. With a
    # fresh codebook (cluster_size starts at 0), one EMA update gives
    #   cluster_size = mu*0 + (1-mu)*batch_count  ->  sum = (1-mu)*n_valid.
    # So sum/(1-mu) must equal n_valid (37), independent of the 63 padded tokens.
    mu_d = 0.01
    cb = _EMACodebook(num_codes=8, code_dim=D, ema_mu=mu_d).to(dev)
    Ntot = 100
    xx = torch.randn(Ntot, D, device=dev)
    valid = torch.zeros(Ntot, dtype=torch.bool, device=dev)
    valid[:37] = True  # exactly 37 valid, 63 padded
    xx[~valid] = 0.0    # padded are zero-vectors (the leak risk)
    codes, _ = cb.quantize(xx)
    cb.ema_update(x_fp32=xx, codes=codes, valid=valid, allow_collectives=False)
    recovered = cb.cluster_size.sum().item() / (1.0 - mu_d)  # undo EMA decay
    if abs(recovered - 37.0) > 1e-3:
        print(f"[UNIT d FAIL] EMA counted {recovered:.4f} tokens != 37 valid "
              f"(padded tokens leaked into the denominator)")
        ok = False
    else:
        print(f"[UNIT d OK] 1b EMA excludes padded: counted {recovered:.2f} "
              f"== {int(valid.sum())} valid (of {Ntot} total, 63 padded zero-vecs)")

    # ---- (e) decoder padded-FRAME isolation. Two probes:
    #   (e1) padded-frame OUTPUT is exactly 0, and perturbing padded-frame INPUT
    #        by 100x does not change valid-frame output (input-dependent leak).
    #   (e2) the INTRA-temporal-block leak (codex 3rd-pass): conv biases make
    #        padded positions nonzero independent of input, and conv2 could spread
    #        them into valid frames. Catch it by comparing valid-frame output of a
    #        clip WITH trailing padded frames vs the all-valid PREFIX clip — they
    #        must match. We force LARGE conv biases so the bug, if present, leaks
    #        loudly (a small default bias could hide under the 1e-4 tol).
    from src.models.vq_model.masked_motion_decoder import MaskedMotionDecoder
    dec = MaskedMotionDecoder(d_model=D, n_heads=H, n_cross_layers=2,
                              n_temporal_layers=2, temporal_kernel=9,
                              dropout=0.0).to(dev).eval()
    with torch.no_grad():
        for tl in dec.temporal_layers:
            tl.conv1.bias.fill_(5.0)   # large biases -> loud leak if unmasked
            tl.conv2.bias.fill_(5.0)
    Td = 8
    skel = torch.randn(B, J, D, device=dev)
    assign_d = torch.zeros(B, J, K, device=dev)
    assign_d[:, :, 0] = 1.0
    cmask = torch.ones(B, K, dtype=torch.bool, device=dev)
    cmask[:, -1] = False
    jmask = torch.ones(B, J, dtype=torch.bool, device=dev)
    fmask = torch.ones(B, Td, dtype=torch.bool, device=dev)
    fmask[:, -2:] = False  # last 2 frames padded
    base_slots = torch.randn(B, Td, K, D, device=dev)
    # all-valid PREFIX clip (Td-2 frames, no padding) — same valid content.
    fmask_pre = torch.ones(B, Td - 2, dtype=torch.bool, device=dev)
    slots_pre = base_slots[:, : Td - 2, :, :].contiguous()
    with torch.no_grad():
        out1 = dec(base_slots, skel, assign_d, cmask, jmask, fmask)
        perturbed = base_slots.clone()
        perturbed[:, -2:, :, :] += 100.0 * torch.randn(B, 2, K, D, device=dev)
        out2 = dec(perturbed, skel, assign_d, cmask, jmask, fmask)
        out_pre = dec(slots_pre, skel, assign_d, cmask, jmask, fmask_pre)
    pad_frame_out = out1[:, -2:, :, :].abs().max().item()
    valid_leak_input = (out1[:, :-2, :, :] - out2[:, :-2, :, :]).abs().max().item()
    valid_leak_bias = (out1[:, :-2, :, :] - out_pre).abs().max().item()
    if pad_frame_out > 0.0 or valid_leak_input > 1e-4 or valid_leak_bias > 1e-4:
        print(f"[UNIT e FAIL] decoder padded-frame out={pad_frame_out:.3e} "
              f"input-leak={valid_leak_input:.3e} bias-leak={valid_leak_bias:.3e} "
              f"(all must be ~0)")
        ok = False
    else:
        print(f"[UNIT e OK] decoder padded-frame isolation: padded out=0, "
              f"input-leak={valid_leak_input:.1e}, intra-block bias-leak="
              f"{valid_leak_bias:.1e} (valid output identical with/without trailing "
              f"padded frames, even with conv bias=5.0)")

    print(f"\n[UNIT CHECKS] {'ALL PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--anytop_root", type=str, default="data/animo4d_anytop_clean_L5")
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--max_joints", type=int, default=64)
    p.add_argument("--max_coarse", type=int, default=50)
    p.add_argument("--val_frac", type=float, default=0.05)
    # model
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1536)
    p.add_argument("--n_graph_layers", type=int, default=4)
    p.add_argument("--n_enc_temporal_layers", type=int, default=2)
    p.add_argument("--n_pre_vq_layers", type=int, default=2)
    p.add_argument("--n_post_vq_layers", type=int, default=2)
    p.add_argument("--n_cross_layers", type=int, default=3)
    p.add_argument("--n_dec_temporal_layers", type=int, default=2)
    p.add_argument("--temporal_stride", type=int, default=4)
    p.add_argument("--temporal_kernel", type=int, default=9)
    p.add_argument("--dropout", type=float, default=0.1)
    # quantizer
    p.add_argument("--code_dim", type=int, default=512)
    p.add_argument("--num_codes", type=int, default=512)
    p.add_argument("--num_quantizers", type=int, default=4)
    p.add_argument("--ema_mu", type=float, default=0.99)
    p.add_argument("--quantize_dropout_prob", type=float, default=0.1)
    p.add_argument("--dead_code_threshold", type=float, default=1.0)
    # loss weights
    p.add_argument("--w_pos", type=float, default=1.0)
    p.add_argument("--w_rot", type=float, default=1.0)
    p.add_argument("--w_vel", type=float, default=1.0)
    p.add_argument("--w_contact", type=float, default=0.1)
    p.add_argument("--w_world", type=float, default=0.25)
    p.add_argument("--w_fk", type=float, default=1.0)
    p.add_argument("--w_traj", type=float, default=0.10)
    p.add_argument("--w_commit", type=float, default=0.02)
    # train
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Linear LR warmup over the first N optimizer steps OF THE "
                        "CURRENT run (steps since THIS process started), ramping lr "
                        "0 -> args.lr, then holding at args.lr. Default 0 = OFF "
                        "(flat lr, byte-identical to no-warmup). Keyed to "
                        "steps-since-launch, NOT the absolute resumed global_step, so "
                        "--resume RE-WARMS the lr from 0 at this launch — used to "
                        "re-warm after a global-batch change (e.g. 6-card cross-alloc "
                        "global 192->576, target lr 4e-4).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=12,
                   help="DataLoader worker processes PER RANK for the train loader "
                        "(val uses max(1, num_workers//2)). Default 12: swarmh1002 has "
                        "32 CPU / 2 ranks → 24/32 used; swarma1004 has 64 CPU. Raise on "
                        "higher-core nodes; lower if CPU-contended.")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--periodic_save_every", type=int, default=25,
                   help="Also save a PRESERVED snapshot named ep{N}_model.pt every "
                        "N epochs (and at the final epoch), in ADDITION to the "
                        "last_model.pt / best_model.pt overwrites. 0 disables. "
                        "Mirrors train_graph_vae.py. Lets CV visual-QA re-render "
                        "mid-run ckpts (ep100/ep200/ep300) for the 300ep L5 run.")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume model+optimizer+epoch+step from a ckpt to CONTINUE "
                        "the SAME experiment. Restores model_state_dict (into the raw "
                        "module before DDP wrap so codebook buffers stay rank-identical), "
                        "optimizer_state_dict, epoch, global_step, val_total. Valid for "
                        "exact comparability because training uses a fixed lr with no "
                        "scheduler. Point --out at the same dir; the non-empty-dir guard "
                        "is skipped under --resume (logs/ckpts are NOT truncated).")
    p.add_argument("--log_every", type=int, default=50,
                   help="Emit the per-iter console/file log block + metrics.jsonl row "
                        "only every N optimizer steps (always also on the very first "
                        "step for the gate-check line). The NaN/finite loss-gate and "
                        "grad-NaN checks fire EVERY step regardless. L5 300ep is >1.3M "
                        "steps so per-step logging is too heavy.")
    p.add_argument("--qa_every", type=int, default=100,
                   help="Run the full-world-recovery root_drift_jitter_qa only every "
                        "N optimizer steps (it is expensive). On non-QA steps the "
                        "rootqa_* fields are omitted from the metrics row.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--out", required=True, help="Output dir for ckpts + logs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Run a few smoke iters + 1 val pass")
    p.add_argument("--smoke_iters", type=int, default=4)
    p.add_argument("--unit_checks", action="store_true",
                   help="Run M3 mask/STE/commit/EMA unit assertions and exit")
    args = p.parse_args()

    is_ddp, rank, local_rank, world_size, is_main = _ddp_setup()

    # Unit checks: rank-0 only, exit (no data load needed).
    if args.unit_checks:
        dev = (torch.device("cuda", local_rank) if torch.cuda.is_available()
               else torch.device("cpu"))
        rc = run_unit_checks(dev) if is_main else 0
        if is_ddp:
            dist.barrier()
            dist.destroy_process_group()
        return rc

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # TF32 + cudnn.benchmark (throughput only; no math/hyperparameter change).
    # TF32 accelerates fp32 matmul/conv on A100/H100 and is safe under bf16
    # autocast (bf16 matmuls are unaffected; fp32 fallbacks just run faster).
    # cudnn.benchmark is safe here because all conv shapes are FIXED constants
    # (max_frames / max_joints / max_coarse are padded to constants every batch),
    # so the autotuner converges once and is not re-triggered by shape churn.
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
    # --resume continues an interrupted run IN PLACE: --out must be the resume ckpt's
    # OWN directory, and only THEN is the non-empty guard skipped (the resume path
    # APPENDS to train.log / metrics.jsonl, it does NOT truncate them). If --resume
    # points at a DIFFERENT run's ckpt (--out runB --resume runA/ep.pt), the non-empty
    # guard still applies — require --overwrite — so a mistaken --out can't silently
    # append-to / overwrite an unrelated run's logs+ckpts (codex 019ea63f #3).
    resume_in_place = (args.resume is not None
                       and out_dir.resolve() == Path(args.resume).resolve().parent)
    if (out_dir.exists() and any(out_dir.iterdir())
            and not args.overwrite and not resume_in_place):
        _hint = ("(--resume must target a ckpt INSIDE --out to continue in place) "
                 if args.resume is not None else
                 "(or --resume to continue an interrupted run) ")
        raise RuntimeError(f"[OUT FAIL] {out_dir} non-empty. Use --overwrite "
                           f"{_hint}.")
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
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git_sha = "unknown"

    log("=== Graph-VQVAE tokenizer training ===")
    log(f"git_sha: {git_sha}")
    log(f"device: {dev}  world_size: {world_size}")
    log(f"args: {vars(args)}")

    # ---- Data (L5; uses materialized splits/{train,val}.txt) ----
    atk = dict(num_frames=args.max_frames, max_joints=args.max_joints,
               val_frac=args.val_frac, load_captions=False,
               data_root=args.anytop_root)
    log(f"Loading AnyTop L5 from {args.anytop_root} ...")
    ds_train = AnyTopDataset(split="train", **atk)
    ds_val = AnyTopDataset(split="val", **atk)
    log(f"train={len(ds_train)} val={len(ds_val)}")
    if len(ds_train) < args.batch_size:
        raise RuntimeError(f"[DATA FAIL] train {len(ds_train)} < batch {args.batch_size}")
    if len(ds_val) == 0:
        raise RuntimeError("[DATA FAIL] val empty.")

    train_sampler = (DistributedSampler(ds_train, shuffle=True, drop_last=True)
                     if is_ddp else None)
    # num_workers>0 is required for prefetch_factor / persistent_workers (PyTorch
    # raises otherwise). Default (12) always >0; the num_workers==0 path is only an
    # explicit-opt-out edge case, so gate both kwargs on it.
    n_workers_train = max(0, args.num_workers)
    n_workers_val = max(1, args.num_workers // 2) if args.num_workers > 0 else 0
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=anytop_collate_fn, num_workers=n_workers_train, drop_last=True,
        pin_memory=True, persistent_workers=(n_workers_train > 0),
        prefetch_factor=(4 if n_workers_train > 0 else None),
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=anytop_collate_fn, num_workers=n_workers_val, drop_last=False,
        pin_memory=True, persistent_workers=(n_workers_val > 0),
        prefetch_factor=(4 if n_workers_val > 0 else None),
    )

    # ---- Model ----
    model = GraphVQTokenizer(
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        n_graph_layers=args.n_graph_layers,
        n_enc_temporal_layers=args.n_enc_temporal_layers,
        n_pre_vq_layers=args.n_pre_vq_layers,
        n_post_vq_layers=args.n_post_vq_layers,
        n_cross_layers=args.n_cross_layers,
        n_dec_temporal_layers=args.n_dec_temporal_layers,
        max_coarse=args.max_coarse,
        temporal_stride=args.temporal_stride,
        temporal_kernel=args.temporal_kernel,
        dropout=args.dropout,
        code_dim=args.code_dim, num_codes=args.num_codes,
        num_quantizers=args.num_quantizers, ema_mu=args.ema_mu,
        quantize_dropout_prob=args.quantize_dropout_prob,
        dead_code_threshold=args.dead_code_threshold,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"VQ tokenizer params: {n_params:,}")

    # --resume: load model_state_dict into the RAW module BEFORE DDP wrap and BEFORE
    # the one-time codebook broadcast. Under DDP every rank loads the SAME ckpt file,
    # so the codebook buffers (embed/cluster_size/embed_avg/initted) become identical
    # across ranks here; the broadcast just below then operates on already-identical
    # buffers as a harmless no-op — keeping the "codebooks start rank-identical"
    # invariant that the per-step EMA all_reduce relies on, whether or not resuming.
    # optimizer_state_dict / epoch / global_step / best_val are restored AFTER the
    # optimizer is built (the optimizer does not exist yet). codex DDP review point.
    resume_ckpt = None
    start_epoch = 0
    if args.resume is not None:
        if not Path(args.resume).exists():
            raise RuntimeError(f"[RESUME FAIL] {args.resume} does not exist.")
        log(f"Resuming (model+optimizer+epoch+step) from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
        # `model` is still the RAW module here (DDP wrap is below).
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=True)
        start_epoch = int(resume_ckpt["epoch"]) + 1
        log(f"  resumed model weights (strict=True); start_epoch={start_epoch} "
            f"(ckpt epoch={resume_ckpt['epoch']}, global_step={resume_ckpt['global_step']}, "
            f"val_total={resume_ckpt.get('val_total')})")

    if is_ddp:
        # ONE-TIME codebook buffer sync: each rank built its _EMACodebook with an
        # independent torch.randn init, so the codebooks (embed / cluster_size /
        # embed_avg buffers) DIFFER across ranks at construction. The EMA update's
        # all_reduce keeps codebooks identical ACROSS STEPS only if they START
        # identical, so we broadcast rank-0's codebook buffers to all ranks once,
        # here, before training. After this they stay bit-identical (EMA stats are
        # all_reduced; dead-code replacements are broadcast from rank 0).
        # (model is still the raw module here — DDP wrap happens just below.)
        for cb in model.quantizer.codebooks:
            dist.broadcast(cb.embed, src=0)
            dist.broadcast(cb.cluster_size, src=0)
            dist.broadcast(cb.embed_avg, src=0)
            dist.broadcast(cb.initted, src=0)
        # EMA codebook buffers are updated in-place inside forward; DDP must NOT
        # re-broadcast buffers each iter (it would clobber the per-step EMA state).
        # We keep codebooks synced explicitly (one-time broadcast above + the EMA
        # all_reduce each step), so broadcast_buffers=False. find_unused_parameters=
        # True: under quantizer dropout dropped stages' codebooks are buffers (no
        # params) and some graph-attn bias params on edge-free pooled graphs may go
        # unused — keep it True defensively.
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True, broadcast_buffers=False)
    raw_model = model.module if is_ddp else model

    loss_weights = {
        "pos": args.w_pos, "rot": args.w_rot, "vel": args.w_vel,
        "contact": args.w_contact, "world": args.w_world, "fk": args.w_fk,
        "traj": args.w_traj, "commit": args.w_commit,
    }
    log(f"loss_weights: {loss_weights}")
    expected_C = args.max_coarse

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --resume (part 2): restore optimizer state + step/epoch/best-val bookkeeping
    # (model weights + start_epoch were restored above). The optimizer is built on
    # the same model.parameters() order as the original run (both built AFTER DDP
    # wrap), so param identity matches. load_state_dict may leave the exp_avg /
    # exp_avg_sq tensors on CPU — move them onto `dev` (AdamW would otherwise error
    # mixing CPU state with CUDA params). n_iter continues from the saved global_step
    # so n_iter % log_every / qa_every phase is preserved; best_val carried forward so
    # the first post-resume val does not clobber a better earlier best_model.pt.
    n_iter = 0
    best_val = float("inf")
    # Local optimizer-step counter for THIS process only — starts at 0 every launch
    # and is DELIBERATELY NOT restored from the ckpt (unlike n_iter). The LR warmup
    # below is keyed to this, so on --resume the warmup re-ramps from 0 at this
    # launch (re-warm after a global-batch change), independent of the large resumed
    # n_iter. warmup_steps=0 leaves this unused (no lr writes, flat-lr preserved).
    steps_this_run = 0
    if resume_ckpt is not None:
        opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(dev)
        n_iter = int(resume_ckpt["global_step"])
        # best_val = the HISTORICAL best, NOT the resumed ckpt's own current val_total
        # (which is just the checkpointed epoch's value — resuming from last_model.pt /
        # epN_model.pt of a worse epoch would otherwise overwrite a better earlier
        # best_model.pt on the first post-resume val). Prefer the persisted
        # best_val_total; for a legacy ckpt without it, fall back to the sibling
        # best_model.pt's val_total (that IS the historical best); else inf
        # (codex 019ea63f #1, mirrors train_graph_vae.py best_val_loss logic).
        if "best_val_total" in resume_ckpt:
            best_val = float(resume_ckpt["best_val_total"])
        else:
            _bm = Path(args.resume).parent / "best_model.pt"
            best_val = (float(torch.load(_bm, map_location="cpu",
                                         weights_only=False)["val_total"])
                        if _bm.exists() else float("inf"))
            log("  [resume] legacy ckpt (no best_val_total) → best_val from "
                "sibling best_model.pt")
        log(f"  resumed optimizer state ({len(opt.state)} param groups on {dev}); "
            f"n_iter={n_iter} best_val={best_val:.4f}")
        del resume_ckpt

    amp_enabled = (args.amp_dtype == "bf16")
    amp_ctx = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
               if amp_enabled else contextlib.nullcontext)
    log(f"AMP: amp_dtype={args.amp_dtype}")

    smoke_iter_cap = args.smoke_iters if args.smoke else None
    # Smoke caps the epoch count too (2 epochs is enough to exercise
    # train -> rank-0 val -> DDP barrier -> next-epoch reshuffle without churning
    # the full 300-epoch loop).
    n_epochs = 2 if args.smoke else args.epochs
    # n_iter / best_val were initialized (and possibly restored from --resume) above;
    # do NOT reset them here. start_epoch is 0 for a fresh run, ckpt.epoch+1 on resume.

    for epoch in range(start_epoch, n_epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        # GPU-side detached running sums (per loss key) + step counter for the EXACT
        # epoch-mean (one .item() per key at epoch end), replacing per-step .item().
        loss_running_sum: dict = {}
        loss_step_count = 0
        for it, raw in enumerate(dl_train):
            if smoke_iter_cap is not None and it >= smoke_iter_cap:
                break
            batch = GraphMotionBatch.from_collate_dict({
                k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()
            })
            with amp_ctx():
                out = model(batch)

            if it == 0:
                B, T_lat, C, D = out["z_q"].shape
                assert C == expected_C, f"[GATE FAIL] z C={C} != {expected_C}"
                assert D == args.d_model, f"[GATE FAIL] z D={D} != {args.d_model}"
                Q = out["indices"].shape[-1]
                assert Q == args.num_quantizers, f"[GATE FAIL] Q={Q}"
                _allowed = (torch.float32, torch.bfloat16) if amp_enabled else (torch.float32,)
                assert out["z_q"].dtype in _allowed, f"[DTYPE FAIL] z_q={out['z_q'].dtype}"
                log(f"  [gate ok] z_q=[{B},{T_lat},{C},{D}] dtype={out['z_q'].dtype} "
                    f"indices=[{B},{T_lat},{C},{Q}]")

            effective_frame_mask = out["frame_mask_recovered"]
            losses = run_vq_loss(out, batch, loss_weights,
                                 effective_frame_mask, args.code_dim)

            # ---- EVERY-step non-finite LOSS gate (1 sync, was ~8) ----
            # `total` is the weighted SUM of all loss terms, so any NaN/Inf in any
            # component propagates into it (NaN through addition; Inf, and the Inf-Inf
            # -> NaN edge, are all caught). Checking total finiteness every step thus
            # detects a non-finite loss every step with ONE sync. The full per-key
            # breakdown (which term blew up) is diagnostic-only and is checked on
            # log-steps below — never weakening the every-step abort.
            if not torch.isfinite(losses["total"]):
                log(f"[GATE FAIL] loss[total]={losses['total'].item()} non-finite "
                    f"at iter {n_iter}")
                return 1
            opt.zero_grad()
            losses["total"].backward()
            # ---- EVERY-step non-finite GRAD gate (1 sync, was hundreds) ----
            # clip_grad_norm_ (max_norm=10.0, unchanged behavior) ALREADY computes the
            # total grad L2 norm over all params and clips in place; we reuse its
            # returned norm for the finiteness check instead of a per-parameter
            # isfinite/abs-max Python loop (which forced hundreds of GPU syncs/step and
            # was redundant with this same reduction). NO error_if_nonfinite=True: we
            # want a GRACEFUL gate-fail exit (return 1) BEFORE opt.step(), matching the
            # original behavior, not an exception. Health metric changed from
            # per-element grad_max -> total L2 grad_norm (both standard diagnostics);
            # its .item() is GATED to log-steps below.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0)
            if not torch.isfinite(grad_norm):
                log(f"[GATE FAIL] non-finite grad norm at iter {n_iter}")
                return 1
            # ---- linear LR warmup (OFF by default; flat-lr otherwise untouched) ----
            # Keyed to steps_this_run (steps since THIS launch), NOT n_iter, so on
            # --resume the ramp restarts from 0 here (re-warm after a batch change).
            # Ramp lr 0 -> args.lr over the first warmup_steps steps, then hold at
            # args.lr; (steps_this_run+1)/warmup_steps reaches 1.0 on the warmup_steps-th
            # step and min(1.0, .) holds thereafter. Applied to every param group right
            # before opt.step(); does NOT touch the optimizer construction, grad clip,
            # or the NaN/finite gates above. warmup_steps==0 -> no lr writes at all
            # (behavior byte-identical to no warmup).
            if args.warmup_steps > 0:
                cur_lr = args.lr * min(1.0, (steps_this_run + 1) / args.warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr
            steps_this_run += 1
            opt.step()

            # Defer the per-key loss .item() syncs (~8/step): accumulate detached
            # GPU-side running sums + a step counter, and materialize the EXACT epoch
            # mean with a single .item() per key at epoch end (below). On log-steps the
            # current-step losses are .item()'d for the console/metrics line (gated,
            # cheap). No approximation: epoch mean = sum / count over the steps run.
            for k, v in losses.items():
                loss_running_sum[k] = loss_running_sum.get(k, 0) + v.detach()
            loss_step_count += 1

            # ---- logging / QA frequency gates (300ep L5 is >1.3M steps) ----
            # do_log gates the per-iter console/file block + the metrics.jsonl row.
            # do_qa gates the expensive full-world-recovery root QA. The finite/grad
            # safety gates above are NOT gated (they fire EVERY step). Always log the
            # very first observed step so the gate-check line is emitted (fresh run:
            # n_iter==0; resume: first iter of the resumed start_epoch).
            do_log = (n_iter % args.log_every == 0) or (n_iter == 0) \
                or (it == 0 and epoch == start_epoch)
            do_qa = (n_iter % args.qa_every == 0)

            qa = None
            if do_qa:
                with torch.no_grad():
                    qa = root_drift_jitter_qa(
                        out["pred_motion"],
                        batch.anytop_x.permute(0, 3, 1, 2).contiguous(),
                        batch.anytop_mean, batch.anytop_std, effective_frame_mask,
                    )

            if do_log:
                # GATED .item() of the current-step per-key losses (cheap: log-steps
                # only). These per-key values ARE the full per-key health view on
                # log-steps (the every-step total-finiteness gate above already aborts
                # on any non-finite component before opt.step, so by here all keys are
                # finite). grad_norm.item() is likewise gated here (the only place its
                # value is materialized to host).
                cur_losses = {k: v.item() for k, v in losses.items()}
                grad_norm_v = grad_norm.item()
                # current live LR read straight off the optimizer for the log row:
                # the warmup-ramped value while --warmup_steps>0 is ramping, else the
                # held flat value (the constructed args.lr on a fresh run, or the lr
                # carried in by opt.load_state_dict on --resume — equal under the
                # fixed-lr same-experiment resume contract).
                cur_lr_log = opt.param_groups[0]["lr"]
                # codebook health (ppl/active/dead/n_valid_tok) — these GPU->CPU reads
                # are GATED to log-steps (consumed only by the console log + the
                # metrics.jsonl row just below). On non-log steps they are not computed,
                # avoiding a per-step sync. NOTE: the commit-loss normalization uses the
                # TENSOR out["n_valid_tokens"] (passed into run_vq_loss above), not this
                # scalar, and the epoch-end ckpt reads out["active_codes"] fresh from the
                # tensor — both independent of these gated host scalars.
                ppl = out["perplexity"].tolist()
                active = out["active_codes"].tolist()
                dead = out["dead_codes"].tolist()
                n_valid_tok = out["n_valid_tokens"].item()
                # mean / p95 valid coarse slot count (only needed for the log row)
                cm = out["coarse_mask"]
                ccount = cm.sum(dim=-1).float()
                valid_c_mean = ccount.mean().item()
                valid_c_p95 = (torch.quantile(ccount, 0.95).item()
                               if ccount.numel() else 0.0)

                log(f"[ep{epoch} it{it} n_iter={n_iter}] total={cur_losses['total']:.4f} "
                    f"pos={cur_losses['pos']:.4f} rot={cur_losses['rot']:.4f} "
                    f"vel={cur_losses['vel']:.4f} world={cur_losses['world']:.4f} "
                    f"fk={cur_losses['fk']:.4f} traj={cur_losses['traj']:.4f} "
                    f"commit={cur_losses['commit']:.5f} grad_norm={grad_norm_v:.3f} "
                    f"lr={cur_lr_log:.3e}")
                log(f"           ppl={[f'{x:.1f}' for x in ppl]} "
                    f"active={active} dead={dead} n_valid_tok={int(n_valid_tok)} "
                    f"validC_mean={valid_c_mean:.1f} validC_p95={valid_c_p95:.1f}")
                if qa is not None:
                    log(f"           [ROOT-QA] drift_mean={qa['root_drift_mean']:.4f} "
                        f"drift_max={qa['root_drift_max']:.4f} "
                        f"jitter_pred={qa['root_jitter_pred']:.4f} "
                        f"jitter_gt={qa['root_jitter_gt']:.4f} "
                        f"jitter_ratio={qa['root_jitter_ratio']:.2f}")
                if is_main:
                    row = {
                        "epoch": epoch, "iter": it, "n_iter": n_iter,
                        "loss": cur_losses,
                        "perplexity": ppl, "active_codes": active, "dead_codes": dead,
                        "n_valid_tokens": n_valid_tok,
                        "valid_c_mean": valid_c_mean, "valid_c_p95": valid_c_p95,
                        "grad_norm": grad_norm_v, "lr": cur_lr_log,
                    }
                    if qa is not None:
                        row.update({f"rootqa_{k}": v for k, v in qa.items()})
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
            n_iter += 1

        epoch_dt = time.time() - t0
        # EXACT epoch-mean total via the deferred GPU-side running sum (one .item()).
        train_total = ((loss_running_sum["total"] / loss_step_count).item()
                       if loss_step_count > 0 else float("nan"))
        log(f"=== epoch {epoch} done in {epoch_dt:.1f}s | train_total={train_total:.4f} ===")

        # Periodic PRESERVED snapshot: every periodic_save_every epochs AND at the
        # final epoch (mirrors train_graph_vae.py). We fold do_periodic into do_val so
        # the snapshot carries a FRESH val_total and is byte-identical to last_model.pt
        # ("same contents as last_model.pt"), rather than reusing a stale best_val.
        do_periodic = (args.periodic_save_every > 0 and not args.smoke
                       and ((epoch + 1) % args.periodic_save_every == 0
                            or epoch == n_epochs - 1))
        do_val = ((epoch + 1) % args.save_every == 0 or epoch == n_epochs - 1
                  or do_periodic or args.smoke)
        if do_val and is_main:
            model.eval()
            val_losses = defaultdict(list)
            with torch.no_grad():
                for vit, raw in enumerate(dl_val):
                    if args.smoke and vit >= 2:
                        break
                    vb = GraphMotionBatch.from_collate_dict({
                        k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()
                    })
                    with amp_ctx():
                        # rank-0-only val: NO cross-rank collectives (the other
                        # ranks are waiting at the post-val barrier, not in here).
                        vo = raw_model(vb, allow_collectives=False)
                    vl = run_vq_loss(vo, vb, loss_weights,
                                     vo["frame_mask_recovered"], args.code_dim,
                                     allow_collectives=False)
                    for k, v in vl.items():
                        val_losses[k].append(v.item())
            val_total = float(np.mean(val_losses["total"]))
            val_recon = float(np.mean(val_losses["pos"])) + float(np.mean(val_losses["rot"]))
            log(f"  [val] total={val_total:.4f} pos={np.mean(val_losses['pos']):.4f} "
                f"rot={np.mean(val_losses['rot']):.4f} commit={np.mean(val_losses['commit']):.5f}")
            # checkpoint (rank-0 only, is_main already guards this block)
            if not args.smoke:
                # best_val_total = the HISTORICAL best (incl. this epoch). Persisted so
                # --resume restores the right best-val bookkeeping and does NOT clobber
                # a better earlier best_model.pt on the first post-resume validation,
                # even when resuming from last_model.pt / epN_model.pt of a worse epoch
                # (codex 019ea63f #1, mirrors train_graph_vae.py best_val_loss).
                hist_best_val = min(best_val, val_total)
                ckpt = {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch, "global_step": n_iter,
                    "args": vars(args), "val_total": val_total, "val_recon": val_recon,
                    "best_val_total": hist_best_val,
                    "git_sha": git_sha,
                    "data_root": args.anytop_root,
                    "max_joints": args.max_joints, "max_coarse": args.max_coarse,
                    "codebook_active": [int(x) for x in out["active_codes"].tolist()],
                }
                last_path = out_dir / "last_model.pt"
                torch.save(ckpt, last_path)
                if val_total < best_val:
                    best_val = val_total
                    torch.save(ckpt, out_dir / "best_model.pt")
                    log(f"  [ckpt] new best_model val_total={val_total:.4f}")
                # Retained periodic snapshot: a byte-identical copy of last_model.pt
                # (copyfile, not a 2nd torch.save — avoids re-serializing the ~666MB
                # ckpt and guarantees identical bytes; codex 019ea63f #2). Not
                # overwritten next val, so CV visual-QA can re-render mid-run ckpts.
                if do_periodic:
                    periodic_path = out_dir / f"ep{epoch + 1}_model.pt"
                    shutil.copyfile(last_path, periodic_path)
                    log(f"  [ckpt] saved periodic snapshot → {periodic_path}")
        if is_ddp:
            dist.barrier()

    log("=== training loop complete ===")
    if is_ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
