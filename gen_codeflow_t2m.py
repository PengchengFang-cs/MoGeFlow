"""Generate HumanML3D motions from text with a released CodeFlow checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from models.codeflow import PartStructuredMotionCodeFlow
from options.codeflow_options import TrainCodeFlowOptions
from train_codeflow import make_config
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric


DEFAULT_REPO_ID = "AmberJar/CodeFlow-HumanML3D"
DEFAULT_CODEFLOW_FILE = "codeflow/codeflow_hml3d_best_top3_ema.pt"
DEFAULT_RVQ_FILE = "rvq/part_vq_hml3d_overlap_best_top3.pth"
DEFAULT_PARTITION_FILE = "rvq/skeleton_partition.json"
DEFAULT_MEAN_FILE = "stats/mean.npy"
DEFAULT_STD_FILE = "stats/std.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--local_dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--vq_checkpoint", type=str, default="")
    parser.add_argument("--vq_partition", type=str, default="")
    parser.add_argument("--mean_path", type=str, default="")
    parser.add_argument("--std_path", type=str, default="")
    parser.add_argument(
        "--clip_path",
        type=str,
        default="",
        help="Optional local OpenAI CLIP ViT-B/32 checkpoint. If omitted, clip.load may download it.",
    )

    parser.add_argument("--text_prompt", type=str, default="")
    parser.add_argument("--text_path", type=str, default="")
    parser.add_argument("--motion_length", type=int, default=196)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--cond_scale", type=float, default=6.0)
    parser.add_argument("--terminal_mode", type=str, default="", choices=["", "nearest", "tied_logits", "learned_head"])
    parser.add_argument("--decode_mode", type=str, default="nearest", choices=["nearest", "ids", "continuous"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="generation/codeflow_hml3d")
    parser.add_argument("--save_mp4", action="store_true")
    return parser.parse_args()


def make_device(args: argparse.Namespace) -> torch.device:
    if args.device:
        device = torch.device(args.device)
    elif args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda", args.gpu_id)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    return device


def hub_download(args: argparse.Namespace, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    local_dir = args.local_dir or None
    return hf_hub_download(
        repo_id=args.repo_id,
        filename=filename,
        revision=args.revision,
        local_dir=local_dir,
    )


def resolve_assets(args: argparse.Namespace) -> Dict[str, str]:
    if args.local_dir:
        root = Path(args.local_dir).expanduser()

        def local_or_download(path_value: str, default_file: str) -> str:
            if path_value:
                return str(Path(path_value).expanduser())
            candidate = root / default_file
            if candidate.is_file():
                return str(candidate)
            return hub_download(args, default_file)

    else:

        def local_or_download(path_value: str, default_file: str) -> str:
            return str(Path(path_value).expanduser()) if path_value else hub_download(args, default_file)

    return {
        "checkpoint": local_or_download(args.checkpoint, DEFAULT_CODEFLOW_FILE),
        "vq_checkpoint": local_or_download(args.vq_checkpoint, DEFAULT_RVQ_FILE),
        "vq_partition": local_or_download(args.vq_partition, DEFAULT_PARTITION_FILE),
        "mean_path": local_or_download(args.mean_path, DEFAULT_MEAN_FILE),
        "std_path": local_or_download(args.std_path, DEFAULT_STD_FILE),
    }


def parse_prompts(args: argparse.Namespace, unit_length: int, max_tokens: int) -> Tuple[List[str], torch.Tensor, List[int]]:
    prompts: List[str] = []
    lengths: List[int] = []
    if args.text_prompt:
        prompts.append(args.text_prompt)
        lengths.append(int(args.motion_length))
    elif args.text_path:
        with Path(args.text_path).expanduser().open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if "#" in line:
                    text, length_text = line.rsplit("#", 1)
                    length = int(length_text) if length_text.isdigit() else int(args.motion_length)
                else:
                    text, length = line, int(args.motion_length)
                prompts.append(text.strip())
                lengths.append(length)
    else:
        raise ValueError("Provide --text_prompt or --text_path")

    if not prompts:
        raise ValueError("No prompts found")
    rounded_lengths = [max(unit_length, int(length) // unit_length * unit_length) for length in lengths]
    token_lengths = [min(max(length // unit_length, 1), max_tokens) for length in rounded_lengths]
    output_lengths = [tokens * unit_length for tokens in token_lengths]
    return prompts, torch.tensor(token_lengths, dtype=torch.long), output_lengths


def load_model(args: argparse.Namespace, assets: Dict[str, str], device: torch.device) -> Tuple[PartStructuredMotionCodeFlow, argparse.Namespace, Dict]:
    ckpt = torch.load(assets["checkpoint"], map_location="cpu", weights_only=False)
    opt = TrainCodeFlowOptions().parser.parse_args([])
    for key, value in ckpt.get("options", {}).items():
        setattr(opt, key, value)

    opt.vq_backend = "kv_part"
    opt.vq_checkpoint = assets["vq_checkpoint"]
    opt.vq_partition = assets["vq_partition"]
    opt.mean_path = assets["mean_path"]
    opt.std_path = assets["std_path"]
    opt.clip_path = str(Path(args.clip_path).expanduser()) if args.clip_path else ""
    opt.kv_root = "."
    opt.gpu_id = args.gpu_id

    model = PartStructuredMotionCodeFlow(make_config(opt))
    state = ckpt["model"]
    missing, unexpected = model.load_trainable_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    model.to(device)
    model.eval()
    return model, opt, ckpt


def maybe_save_mp4(path: Path, joints: np.ndarray, caption: str) -> None:
    from utils.paramUtil import t2m_kinematic_chain
    from utils.plot_script import plot_3d_motion

    plot_3d_motion(str(path), t2m_kinematic_chain, joints, title=caption, fps=20)


def main() -> None:
    args = parse_args()
    if args.repeat_times <= 0:
        raise ValueError("--repeat_times must be positive")

    device = make_device(args)
    assets = resolve_assets(args)
    model, opt, ckpt = load_model(args, assets, device)

    unit_length = int(getattr(opt, "unit_length", 4))
    max_tokens = int(getattr(opt, "max_motion_tokens", 49))
    prompts, token_lengths_cpu, output_lengths = parse_prompts(args, unit_length, max_tokens)
    token_lengths = token_lengths_cpu.to(device)

    mean = np.load(assets["mean_path"]).astype(np.float32)
    std = np.load(assets["std_path"]).astype(np.float32)
    output_dir = Path(args.output_dir).expanduser()
    feature_dir = output_dir / "features"
    joint_dir = output_dir / "joints"
    id_dir = output_dir / "ids"
    video_dir = output_dir / "animations"
    for path in (feature_dir, joint_dir, id_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_mp4:
        video_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for repeat_id in range(args.repeat_times):
        fixseed(int(args.seed) + repeat_id)
        with torch.no_grad():
            motion_norm, ids = model.generate_motion(
                prompts,
                token_lengths=token_lengths,
                steps=int(args.steps),
                cond_scale=float(args.cond_scale),
                terminal_mode=args.terminal_mode or getattr(opt, "terminal_mode", None),
                decode_mode=args.decode_mode,
            )
        motion_norm_np = motion_norm.detach().cpu().numpy()
        ids_np = ids.detach().cpu().numpy()
        motion_hml3d_np = motion_norm_np * std[None, None, :] + mean[None, None, :]

        for sample_id, caption in enumerate(prompts):
            length = output_lengths[sample_id]
            stem = f"sample{sample_id:03d}_repeat{repeat_id:02d}_len{length}"
            feature_norm_path = feature_dir / f"{stem}_normalized.npy"
            feature_path = feature_dir / f"{stem}_hml3d.npy"
            joint_path = joint_dir / f"{stem}_joints.npy"
            ids_path = id_dir / f"{stem}_ids.npy"

            features = motion_hml3d_np[sample_id, :length]
            joints = recover_from_ric(torch.from_numpy(features).float(), 22).numpy()
            np.save(feature_norm_path, motion_norm_np[sample_id, :length])
            np.save(feature_path, features)
            np.save(joint_path, joints)
            np.save(ids_path, ids_np[sample_id, : token_lengths_cpu[sample_id].item()])
            record = {
                "caption": caption,
                "repeat": repeat_id,
                "length": int(length),
                "normalized_features": str(feature_norm_path),
                "humanml3d_features": str(feature_path),
                "joints": str(joint_path),
                "ids": str(ids_path),
            }
            if args.save_mp4:
                video_path = video_dir / f"{stem}.mp4"
                maybe_save_mp4(video_path, joints, caption)
                record["mp4"] = str(video_path)
            records.append(record)
            print(f"saved {stem}: {caption}")

    payload = {
        "repo_id": args.repo_id,
        "checkpoint": assets["checkpoint"],
        "rvq_checkpoint": assets["vq_checkpoint"],
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("step"),
        "weight_source": ckpt.get("weight_source", "model"),
        "steps": int(args.steps),
        "cond_scale": float(args.cond_scale),
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
