#!/usr/bin/env python3
"""Evaluate trained KV-Control part-aware VQ-VAE checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_kv_part_vq import build_model, evaluate_retrieval, load_eval_components, set_seed

# Standalone VQ checkpoint metrics are always reported on the test split,
# regardless of the eval_split stored in the training-time checkpoint config.
EVAL_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    parser.add_argument("--dataset_name", choices=["kit", "t2m"], default="kit")
    parser.add_argument("--data_root", type=Path, default=Path("dataset/KIT-ML"))
    parser.add_argument("--kv_root", type=Path, default=None)
    parser.add_argument("--partition_file", type=Path, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_seed", type=int, default=3407)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_model_args(cli_args: argparse.Namespace, checkpoint_config: dict) -> SimpleNamespace:
    cfg = dict(checkpoint_config)
    cfg["eval_split"] = EVAL_SPLIT
    cfg["dataset_name"] = cli_args.dataset_name or cfg.get("dataset_name", "kit")
    cfg["data_root"] = str(cli_args.data_root.resolve())
    if cli_args.kv_root is not None:
        cfg["kv_root"] = str(cli_args.kv_root.resolve())
    if cli_args.partition_file is not None:
        cfg["partition_file"] = str(cli_args.partition_file.resolve())
    cfg["eval_batch_size"] = cli_args.eval_batch_size

    required_defaults = {
        "code_dim": 128,
        "nb_code": 128,
        "mu": 0.99,
        "down_t": 2,
        "stride_t": 2,
        "width": 512,
        "depth": 3,
        "dilation_growth_rate": 3,
        "output_emb_width": 128,
        "vq_act": "relu",
        "vq_norm": None,
        "quantizer": "ema_reset",
    }
    for key, value in required_defaults.items():
        cfg.setdefault(key, value)

    for key in ("data_root", "kv_root", "partition_file", "output_dir"):
        if cfg.get(key) is not None:
            cfg[key] = Path(cfg[key]).resolve()
    return SimpleNamespace(**cfg)


def evaluator_artifacts(dataset_name: str) -> dict[str, Path]:
    root = Path("checkpoints") / dataset_name
    return {
        "opt": root / "Comp_v6_KLD005" / "opt.txt",
        "mean": root / "Comp_v6_KLD005" / "meta" / "mean.npy",
        "std": root / "Comp_v6_KLD005" / "meta" / "std.npy",
        "checkpoint": root / "text_mot_match" / "model" / "finest.tar",
    }


def evaluate_checkpoint(ckpt_path: Path, cli_args: argparse.Namespace, device: torch.device) -> dict:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = make_model_args(cli_args, checkpoint.get("config", {}))
    model = build_model(model_args, device)
    state_key = "vq_model"
    state = checkpoint.get(state_key)
    if state is None:
        state_key = "net"
        state = checkpoint.get(state_key)
    if state is None:
        raise KeyError(f"`vq_model` or `net` not found in checkpoint: {ckpt_path}")
    model.load_state_dict(state)
    model.eval()

    eval_loader, eval_wrapper = load_eval_components(model_args, device)

    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        set_seed(int(cli_args.eval_seed))
        metrics = evaluate_retrieval(model, eval_loader, eval_wrapper, device)
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    artifacts = evaluator_artifacts(model_args.dataset_name)
    artifact_info = {}
    for key, path in artifacts.items():
        artifact_info[key] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path) if path.exists() else None,
            "exists": path.exists(),
        }

    return {
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_state_key": state_key,
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_epoch": int(checkpoint.get("ep", -1)),
        "dataset_name": model_args.dataset_name,
        "eval_split": EVAL_SPLIT,
        "eval_seed": cli_args.eval_seed,
        "eval_batch_size": cli_args.eval_batch_size,
        "evaluator": artifact_info,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if not torch.cuda.is_available() and args.gpu_id >= 0:
        raise RuntimeError("CUDA is not available in this process")
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    results = [evaluate_checkpoint(path.resolve(), args, device) for path in args.checkpoint]
    payload = {"results": results}

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
