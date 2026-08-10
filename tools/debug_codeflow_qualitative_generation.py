#!/usr/bin/env python3
"""Generate small CodeFlow qualitative diagnostics with explicit weight source."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gen_codeflow_t2m import load_model, parse_prompts, resolve_assets
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", nargs=3, metavar=("NAME", "PATH", "SOURCE"), required=True)
    parser.add_argument("--vq_checkpoint", required=True)
    parser.add_argument("--vq_partition", required=True)
    parser.add_argument("--mean_path", required=True)
    parser.add_argument("--std_path", required=True)
    parser.add_argument("--clip_path", required=True)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--cond_scale", type=float, default=6.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


def base_namespace(args: argparse.Namespace, checkpoint_path: str, output_dir: Path) -> Namespace:
    return Namespace(
        repo_id="AmberJar/CodeFlow-HumanML3D",
        revision="main",
        local_dir="",
        checkpoint=str(checkpoint_path),
        vq_checkpoint=str(args.vq_checkpoint),
        vq_partition=str(args.vq_partition),
        mean_path=str(args.mean_path),
        std_path=str(args.std_path),
        clip_path=str(args.clip_path),
        text_prompt="",
        text_path=str(args.prompt_path),
        motion_length=196,
        repeat_times=1,
        steps=int(args.steps),
        cond_scale=float(args.cond_scale),
        terminal_mode="",
        seed=int(args.seed),
        gpu_id=int(args.gpu_id),
        device=f"cuda:{int(args.gpu_id)}" if torch.cuda.is_available() else "cpu",
        output_dir=str(output_dir),
        save_mp4=False,
    )


def save_variant(args: argparse.Namespace, name: str, checkpoint_path: str, source: str) -> None:
    source = source.lower()
    if source not in {"model", "ema"}:
        raise ValueError(f"Unsupported source {source!r}; expected model or ema")

    output_dir = args.output_root / name
    gen_args = base_namespace(args, checkpoint_path, output_dir)
    device = torch.device(gen_args.device)
    assets = resolve_assets(gen_args)
    model, opt, ckpt = load_model(gen_args, assets, device)
    if source == "ema":
        missing, unexpected = model.load_trainable_state_dict(ckpt["ema"])
        if missing or unexpected:
            raise RuntimeError(f"EMA load mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")

    model.eval()
    unit_length = int(getattr(opt, "unit_length", 4))
    max_tokens = int(getattr(opt, "max_motion_tokens", 49))
    captions, token_lengths_cpu, output_lengths = parse_prompts(gen_args, unit_length, max_tokens)
    token_lengths = token_lengths_cpu.to(device)

    mean = np.load(assets["mean_path"]).astype(np.float32)
    std = np.load(assets["std_path"]).astype(np.float32)
    for subdir in ("features", "joints", "ids"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    fixseed(int(args.seed))
    with torch.no_grad():
        motion_norm, ids = model.generate_motion(
            captions,
            token_lengths=token_lengths,
            steps=int(args.steps),
            cond_scale=float(args.cond_scale),
            terminal_mode=getattr(opt, "terminal_mode", None),
        )

    motion_norm_np = motion_norm.detach().cpu().numpy()
    ids_np = ids.detach().cpu().numpy()
    motion_hml3d_np = motion_norm_np * std[None, None, :] + mean[None, None, :]

    records = []
    for sample_id, caption in enumerate(captions):
        length = int(output_lengths[sample_id])
        stem = f"sample{sample_id:03d}_repeat00_len{length}"
        features = motion_hml3d_np[sample_id, :length]
        joints = recover_from_ric(torch.from_numpy(features).float(), 22).numpy()
        feature_path = output_dir / "features" / f"{stem}_hml3d.npy"
        feature_norm_path = output_dir / "features" / f"{stem}_normalized.npy"
        joint_path = output_dir / "joints" / f"{stem}_joints.npy"
        ids_path = output_dir / "ids" / f"{stem}_ids.npy"
        np.save(feature_path, features)
        np.save(feature_norm_path, motion_norm_np[sample_id, :length])
        np.save(joint_path, joints)
        np.save(ids_path, ids_np[sample_id, : token_lengths_cpu[sample_id].item()])
        records.append(
            {
                "caption": caption,
                "repeat": 0,
                "length": length,
                "humanml3d_features": str(feature_path),
                "normalized_features": str(feature_norm_path),
                "joints": str(joint_path),
                "ids": str(ids_path),
            }
        )
        print(f"saved {name} {stem}: {caption}", flush=True)

    payload = {
        "checkpoint": str(checkpoint_path),
        "weight_source": source,
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("step"),
        "steps": int(args.steps),
        "cond_scale": float(args.cond_scale),
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    for name, checkpoint_path, source in args.checkpoint:
        print(f"=== {name} {source} {checkpoint_path}", flush=True)
        save_variant(args, name, checkpoint_path, source)


if __name__ == "__main__":
    main()
