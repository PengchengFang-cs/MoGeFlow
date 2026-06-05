"""Run full HumanML3D text-to-motion evaluation for CodeFlow checkpoints."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Type

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.t2m_dataset import Text2MotionDatasetEval, collate_fn
from models.codeflow import CodeFlowEvalConfig, MotionCodeFlow, evaluate_codeflow_t2m
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from options.codeflow_options import TrainCodeFlowOptions
from train_codeflow import make_config
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.word_vectorizer import WordVectorizer


SCALAR_METRICS = (
    "fid",
    "diversity",
    "diversity_real",
    "top1",
    "top2",
    "top3",
    "matching_score",
    "matching_score_real",
    "multimodality",
    "code_token_acc",
    "code_wrong_rate",
    "geom_code_dist",
    "geom_rank_pct",
    "geom_wrong_code_dist",
    "geom_wrong_rank_pct",
    "geom_severe_rate",
    "geom_wrong_severe_frac",
    "nb_sample",
)
VECTOR_METRICS = ("r_precision", "r_precision_real")
EVAL_SPLIT = "test"


def parse_args(defaults: Dict[str, object] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="", help="Path to a CodeFlow .pt checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/t2m/codeflow_tconcat_dit_b")
    parser.add_argument("--which_epoch", type=str, default="latest")
    parser.add_argument("--eval_dir", type=str, default="")

    parser.add_argument("--dataset_opt_path", type=str, default="./checkpoints/t2m/Comp_v6_KLD005/opt.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--repeat_times", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10107)
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--device", type=str, default="")

    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cond_scale", type=float, default=3.0)
    parser.add_argument("--terminal_mode", type=str, default="", choices=["", "nearest", "tied_logits", "learned_head"])
    parser.add_argument("--decode_mode", type=str, default="", choices=["", "nearest", "ids", "continuous"])
    parser.add_argument("--unit_length", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--disable_mm", action="store_true")
    parser.add_argument("--mm_num_batches", type=int, default=3)
    parser.add_argument("--mm_num_samples", type=int, default=30)
    parser.add_argument("--multimodality_times", type=int, default=10)
    parser.add_argument("--allow_small_eval", action="store_true", help="Only for smoke tests with tiny max_batches.")
    parser.add_argument("--disable_code_metrics", action="store_true")
    parser.add_argument("--geometry_severe_quantile", type=float, default=0.75)
    parser.add_argument("--no_ema", action="store_true", help="Use raw model weights even when checkpoint EMA exists.")

    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--glove_dir", type=str, default="./glove")
    parser.add_argument("--kv_root", type=str, default="")
    parser.add_argument("--vq_checkpoint", type=str, default="")
    parser.add_argument("--vq_partition", type=str, default="")
    parser.add_argument("--mean_path", type=str, default="")
    parser.add_argument("--std_path", type=str, default="")
    parser.add_argument("--clip_path", type=str, default="")
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args()


def fill_artifact_defaults(opt: Namespace) -> Namespace:
    kv_root = Path(opt.kv_root).expanduser()
    if not getattr(opt, "vq_checkpoint", ""):
        opt.vq_checkpoint = str(kv_root / "checkpoints" / "vqvae" / "net_best_fid.pth")
    if not getattr(opt, "vq_partition", ""):
        opt.vq_partition = str(kv_root / "checkpoints" / "vqvae" / "skeleton_partition.json")
    if not getattr(opt, "mean_path", ""):
        opt.mean_path = str(kv_root / "checkpoints" / "stats" / "mean.npy")
    if not getattr(opt, "std_path", ""):
        opt.std_path = str(kv_root / "checkpoints" / "stats" / "std.npy")
    if not getattr(opt, "clip_path", ""):
        opt.clip_path = str(kv_root / "checkpoints" / "clip" / "ViT-B-32.pt")
    return opt


def default_train_options() -> Namespace:
    opt = TrainCodeFlowOptions().parser.parse_args([])
    return fill_artifact_defaults(opt)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path.resolve()

    model_dir = Path(args.output_dir).expanduser() / "model"
    which = args.which_epoch
    if which == "latest":
        candidates = [model_dir / "latest.pt", model_dir / "latest.tar"]
    elif Path(which).suffix:
        candidates = [model_dir / which]
    elif which.isdigit():
        candidates = [model_dir / f"step_{int(which):08d}.pt"]
    else:
        candidates = [model_dir / f"{which}.pt"]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("checkpoint not found; tried: " + ", ".join(str(c) for c in candidates))


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


def apply_checkpoint_and_cli_options(base: Namespace, ckpt_options: Dict[str, object], args: argparse.Namespace) -> Namespace:
    for key, value in ckpt_options.items():
        setattr(base, key, value)

    for attr in ("kv_root", "vq_checkpoint", "vq_partition", "mean_path", "std_path", "clip_path", "data_root"):
        value = getattr(args, attr)
        if value:
            setattr(base, attr, value)
    if args.unit_length > 0:
        base.unit_length = args.unit_length
    base.gpu_id = args.gpu_id
    return fill_artifact_defaults(base)


def load_codeflow_model(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    model_cls: Type[MotionCodeFlow] = MotionCodeFlow,
) -> Tuple[MotionCodeFlow, Namespace, Dict[str, object], str]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    opt = apply_checkpoint_and_cli_options(default_train_options(), ckpt.get("options", {}), args)
    model = model_cls(make_config(opt))

    use_ema = (not args.no_ema) and ckpt.get("ema") is not None
    state = ckpt["ema"] if use_ema else ckpt["model"]
    missing, unexpected = model.load_trainable_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")

    model.to(device)
    model.eval()
    return model, opt, ckpt, "ema" if use_ema else "model"


def build_eval_loader(
    args: argparse.Namespace,
    opt: Namespace,
    device: torch.device,
) -> Tuple[DataLoader, Text2MotionDatasetEval, Namespace]:
    eval_opt = get_opt(args.dataset_opt_path, device)
    eval_opt.checkpoints_dir = args.checkpoints_dir
    eval_opt.save_root = str(Path(eval_opt.checkpoints_dir) / eval_opt.dataset_name / eval_opt.name)
    eval_opt.model_dir = str(Path(eval_opt.save_root) / "model")
    eval_opt.meta_dir = str(Path(eval_opt.save_root) / "meta")

    data_root = args.data_root or getattr(opt, "data_root", "") or eval_opt.data_root
    eval_opt.data_root = data_root
    eval_opt.motion_dir = str(Path(data_root) / "new_joint_vecs")
    eval_opt.text_dir = str(Path(data_root) / "texts")
    eval_opt.unit_length = int(getattr(opt, "unit_length", 4))

    mean = np.load(str(Path(eval_opt.meta_dir) / "mean.npy")).astype(np.float32)
    std = np.load(str(Path(eval_opt.meta_dir) / "std.npy")).astype(np.float32)
    split_file = str(Path(eval_opt.data_root) / f"{EVAL_SPLIT}.txt")
    w_vectorizer = WordVectorizer(args.glove_dir, "our_vab")
    dataset = Text2MotionDatasetEval(eval_opt, mean, std, split_file, w_vectorizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
    )
    return loader, dataset, eval_opt


def aggregate_metrics(repeat_metrics: Iterable[Dict[str, object]]) -> Dict[str, object]:
    metrics = list(repeat_metrics)
    out: Dict[str, object] = {}
    repeat_count = len(metrics)
    for key in SCALAR_METRICS:
        if key not in metrics[0]:
            continue
        values = np.asarray([float(m[key]) for m in metrics], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()),
            "conf95": float(1.96 * values.std(ddof=0) / np.sqrt(max(repeat_count, 1))),
            "values": values.tolist(),
        }
    for key in VECTOR_METRICS:
        if key not in metrics[0]:
            continue
        values = np.asarray([m[key] for m in metrics], dtype=np.float64)
        out[key] = {
            "mean": values.mean(axis=0).astype(float).tolist(),
            "conf95": (1.96 * values.std(axis=0, ddof=0) / np.sqrt(max(repeat_count, 1))).astype(float).tolist(),
            "values": values.astype(float).tolist(),
        }
    return out


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_results(
    args: argparse.Namespace,
    checkpoint_path: Path,
    opt: Namespace,
    eval_cfg: CodeFlowEvalConfig,
    weight_source: str,
    repeat_metrics: List[Dict[str, object]],
) -> Path:
    eval_dir = Path(args.eval_dir) if args.eval_dir else Path(getattr(opt, "output_dir", args.output_dir)) / "eval_codeflow"
    eval_dir.mkdir(parents=True, exist_ok=True)
    terminal = eval_cfg.terminal_mode or getattr(opt, "terminal_mode", "checkpoint")
    decode = eval_cfg.decode_mode or getattr(opt, "decode_mode", "nearest")
    stem = f"{checkpoint_path.stem}_{EVAL_SPLIT}_s{eval_cfg.steps}_cfg{eval_cfg.cond_scale:g}_{terminal}_{decode}"
    path = eval_dir / f"{stem}.json"
    payload = {
        "checkpoint": str(checkpoint_path),
        "split": EVAL_SPLIT,
        "weight_source": weight_source,
        "repeat_times": len(repeat_metrics),
        "args": vars(args),
        "train_options": vars(opt),
        "eval_config": vars(eval_cfg),
        "repeat_metrics": repeat_metrics,
        "aggregate": aggregate_metrics(repeat_metrics),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    return path


def print_summary(aggregate: Dict[str, object]) -> None:
    fid = aggregate["fid"]
    top1 = aggregate["top1"]
    top2 = aggregate["top2"]
    top3 = aggregate["top3"]
    matching = aggregate["matching_score"]
    diversity = aggregate["diversity"]
    mm = aggregate["multimodality"]
    code_acc = aggregate.get("code_token_acc")
    print(
        "FINAL CodeFlow Eval: "
        f"FID {fid['mean']:.4f} +- {fid['conf95']:.4f}, "
        f"Top1 {top1['mean']:.4f} +- {top1['conf95']:.4f}, "
        f"Top2 {top2['mean']:.4f} +- {top2['conf95']:.4f}, "
        f"Top3 {top3['mean']:.4f} +- {top3['conf95']:.4f}, "
        f"Matching {matching['mean']:.4f} +- {matching['conf95']:.4f}, "
        f"Diversity {diversity['mean']:.4f} +- {diversity['conf95']:.4f}, "
        f"MM {mm['mean']:.4f} +- {mm['conf95']:.4f}"
        + (f", CodeAcc {code_acc['mean']:.4f} +- {code_acc['conf95']:.4f}" if code_acc is not None else "")
    )


def main(
    model_cls: Type[MotionCodeFlow] = MotionCodeFlow,
    parser_defaults: Dict[str, object] = None,
) -> None:
    args = parse_args(parser_defaults)
    if args.repeat_times <= 0:
        raise ValueError("--repeat_times must be positive")

    checkpoint_path = resolve_checkpoint(args)
    device = make_device(args)
    model, opt, _ckpt, weight_source = load_codeflow_model(checkpoint_path, args, device, model_cls=model_cls)
    loader, dataset, eval_opt = build_eval_loader(args, opt, device)
    eval_wrapper = EvaluatorModelWrapper(eval_opt)

    vq_mean = np.load(opt.mean_path).astype(np.float32)
    vq_std = np.load(opt.std_path).astype(np.float32)
    eval_cfg = CodeFlowEvalConfig(
        steps=args.steps,
        cond_scale=args.cond_scale,
        terminal_mode=args.terminal_mode or None,
        decode_mode=args.decode_mode or getattr(opt, "decode_mode", None),
        unit_length=int(getattr(opt, "unit_length", 4)),
        max_batches=args.max_batches,
        cal_mm=not args.disable_mm,
        mm_num_batches=args.mm_num_batches,
        mm_num_samples=args.mm_num_samples,
        multimodality_times=args.multimodality_times,
        allow_small_eval=args.allow_small_eval,
        include_code_metrics=not args.disable_code_metrics,
        geometry_severe_quantile=args.geometry_severe_quantile,
    )

    print(
        f"Evaluating {checkpoint_path} on {EVAL_SPLIT} with {weight_source} weights, "
        f"device={device}, dataset_size={len(dataset)}, repeats={args.repeat_times}"
    )

    repeat_metrics: List[Dict[str, object]] = []
    for repeat_id in range(args.repeat_times):
        fixseed(args.seed + repeat_id)
        metrics = evaluate_codeflow_t2m(
            loader=loader,
            model=model,
            eval_wrapper=eval_wrapper,
            vq_mean=vq_mean,
            vq_std=vq_std,
            eval_mean=dataset.mean,
            eval_std=dataset.std,
            cfg=eval_cfg,
            repeat_id=repeat_id,
        )
        repeat_metrics.append(metrics)

    result_path = write_results(args, checkpoint_path, opt, eval_cfg, weight_source, repeat_metrics)
    aggregate = aggregate_metrics(repeat_metrics)
    print_summary(aggregate)
    print(f"Saved eval results to {result_path}")


if __name__ == "__main__":
    main()
