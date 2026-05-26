#!/usr/bin/env python3
"""Train the KV-Control part-aware VQ-VAE on HumanML3D or KIT features."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset_name", choices=["kit", "t2m"], default="kit")
    parser.add_argument("--data_root", type=Path, default=Path("dataset/KIT-ML"))
    parser.add_argument(
        "--kv_root",
        type=Path,
        default=None,
        help="Optional external KV-Control checkout. Omit to use the vendored kvctrl package in this repo.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--partition_file", type=Path, default=Path("configs/kit_skeleton_partition_pscf.json"))

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=32)

    parser.add_argument("--total_iter", type=int, default=300000)
    parser.add_argument("--warm_up_iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--milestones", nargs="+", type=int, default=[200000])
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--commit", type=float, default=0.02)
    parser.add_argument("--loss_vel", type=float, default=0.5)
    parser.add_argument("--recons_loss", choices=["l1", "l1_smooth"], default="l1_smooth")

    parser.add_argument("--code_dim", type=int, default=128)
    parser.add_argument("--nb_code", type=int, default=128)
    parser.add_argument("--mu", type=float, default=0.99)
    parser.add_argument("--down_t", type=int, default=2)
    parser.add_argument("--stride_t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation_growth_rate", type=int, default=3)
    parser.add_argument("--output_emb_width", type=int, default=128)
    parser.add_argument("--vq_act", choices=["relu", "silu", "gelu"], default="relu")
    parser.add_argument("--vq_norm", default=None)
    parser.add_argument("--quantizer", choices=["ema_reset", "orig", "ema", "reset"], default="ema_reset")

    parser.add_argument("--feat_bias", type=float, default=5.0)
    parser.add_argument(
        "--stat_bias_passes",
        type=int,
        default=2,
        help="Number of times to apply feat_bias to root/contact std. 2 matches the existing KIT VQ/evaluator metadata.",
    )
    parser.add_argument("--print_iter", type=int, default=200)
    parser.add_argument("--eval_iter", type=int, default=5000)
    parser.add_argument("--save_latest", type=int, default=500)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--disable_eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gpu_id", type=int, default=0)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "root": args.output_dir,
        "model": args.output_dir / "model",
        "meta": args.output_dir / "meta",
        "stats": args.output_dir / "stats",
        "logs": args.output_dir / "logs",
        "config": args.output_dir / "config",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def dataset_shape(dataset_name: str) -> tuple[int, int]:
    if dataset_name == "kit":
        return 251, 21
    if dataset_name == "t2m":
        return 263, 22
    raise ValueError(f"Unknown dataset: {dataset_name}")


def apply_feat_bias(std: np.ndarray, joints_num: int, feat_bias: float, passes: int) -> np.ndarray:
    std = std.copy()
    for _ in range(passes):
        std[0:1] = std[0:1] / feat_bias
        std[1:3] = std[1:3] / feat_bias
        std[3:4] = std[3:4] / feat_bias
        std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
        std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = (
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] / 1.0
        )
        std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = (
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] / 1.0
        )
        std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] / feat_bias
        )
    return std


def load_stats(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    dim_pose, joints_num = dataset_shape(args.dataset_name)
    mean = np.load(args.data_root / "Mean.npy").astype(np.float32)
    std_raw = np.load(args.data_root / "Std.npy").astype(np.float32)

    expected_dim = 4 + (joints_num - 1) * 9 + joints_num * 3 + 4
    if mean.shape[-1] != dim_pose or std_raw.shape[-1] != dim_pose or expected_dim != dim_pose:
        raise ValueError(
            f"Unexpected feature dim: mean={mean.shape}, std={std_raw.shape}, expected={dim_pose}"
        )

    std = apply_feat_bias(std_raw, joints_num, args.feat_bias, args.stat_bias_passes).astype(np.float32)
    for out_dir in (paths["meta"], paths["stats"]):
        np.save(out_dir / "mean.npy", mean)
        np.save(out_dir / "std.npy", std)
    return mean, std


class FixedWindowMotionDataset(Dataset):
    def __init__(self, data_root: Path, split: str, window_size: int, mean: np.ndarray, std: np.ndarray) -> None:
        self.motion_dir = data_root / "new_joint_vecs"
        self.window_size = window_size
        self.mean = mean
        self.std = std
        self.data: list[np.ndarray] = []
        self.lengths: list[int] = []

        split_path = data_root / f"{split}.txt"
        ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        missing = 0
        short = 0
        for name in ids:
            motion_path = self.motion_dir / f"{name}.npy"
            if not motion_path.exists():
                missing += 1
                continue
            motion = np.load(motion_path).astype(np.float32)
            if motion.shape[0] < window_size:
                short += 1
                continue
            self.lengths.append(motion.shape[0] - window_size)
            self.data.append(motion)

        self.cumsum = np.cumsum([0] + self.lengths)
        print(
            f"[dataset:{split}] motions={len(self.data)} snippets={int(self.cumsum[-1])} "
            f"missing={missing} short={short}"
        )
        if int(self.cumsum[-1]) <= 0:
            raise RuntimeError(f"No snippets loaded for split {split}")

    def __len__(self) -> int:
        return int(self.cumsum[-1])

    def __getitem__(self, item: int) -> torch.Tensor:
        if item != 0:
            motion_id = int(np.searchsorted(self.cumsum, item) - 1)
            idx = int(item - self.cumsum[motion_id] - 1)
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.window_size]
        motion = (motion - self.mean) / self.std
        return torch.from_numpy(motion.astype(np.float32))


def build_model(args: argparse.Namespace, device: torch.device):
    if args.kv_root is not None:
        sys.path.insert(0, str(args.kv_root))
    from kvctrl.models.vqvae import HumanVQVAE

    dim_pose, _ = dataset_shape(args.dataset_name)
    model_args = SimpleNamespace(
        dataname=args.dataset_name,
        input_dim=dim_pose,
        quantizer=args.quantizer,
        mu=args.mu,
        partition_file=str(args.partition_file.resolve()) if args.partition_file else None,
    )
    model = HumanVQVAE(
        model_args,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act,
        norm=args.vq_norm,
    ).to(device)
    return model


def reconstruction_loss(kind: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "l1_smooth":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(kind)


def compute_losses(args: argparse.Namespace, model, motions: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    _, joints_num = dataset_shape(args.dataset_name)
    pred_motion, loss_commit, perplexity = model(motions)
    loss_rec = reconstruction_loss(args.recons_loss, pred_motion, motions)
    pos_end = 4 + (joints_num - 1) * 3
    loss_explicit = reconstruction_loss(args.recons_loss, pred_motion[..., 4:pos_end], motions[..., 4:pos_end])
    loss = loss_rec + args.loss_vel * loss_explicit + args.commit * loss_commit
    return OrderedDict(
        loss=loss,
        loss_rec=loss_rec,
        loss_explicit=loss_explicit,
        loss_commit=loss_commit,
        perplexity=perplexity,
    )


def lr_at_step(args: argparse.Namespace, step: int) -> float:
    if args.warm_up_iter > 0 and step < args.warm_up_iter:
        return args.lr * float(step + 1) / float(args.warm_up_iter + 1)
    decays = sum(step >= milestone for milestone in args.milestones)
    return args.lr * (args.gamma ** decays)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def average_val_loss(args: argparse.Namespace, model, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: OrderedDict[str, float] = OrderedDict()
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            motions = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            losses = compute_losses(args, model, motions)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
            count += 1
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_retrieval(model, eval_loader: DataLoader, eval_wrapper, device: torch.device) -> dict[str, float]:
    from utils.metrics import (
        calculate_R_precision,
        calculate_activation_statistics,
        calculate_diversity,
        calculate_frechet_distance,
        euclidean_distance_matrix,
    )

    model.eval()
    motion_annotation_list = []
    motion_pred_list = []
    r_precision_real = 0.0
    r_precision = 0.0
    matching_score_real = 0.0
    matching_score_pred = 0.0
    nb_sample = 0

    for batch in eval_loader:
        word_embeddings, pos_one_hots, _caption, sent_len, motion, m_length, _token = batch
        motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        pred_pose_eval, _loss_commit, _perplexity = model(motion)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        et_np = et.detach().cpu().numpy()
        em_np = em.detach().cpu().numpy()
        et_pred_np = et_pred.detach().cpu().numpy()
        em_pred_np = em_pred.detach().cpu().numpy()
        r_precision_real += calculate_R_precision(et_np, em_np, top_k=3, sum_all=True)
        matching_score_real += euclidean_distance_matrix(et_np, em_np).trace()
        r_precision += calculate_R_precision(et_pred_np, em_pred_np, top_k=3, sum_all=True)
        matching_score_pred += euclidean_distance_matrix(et_pred_np, em_pred_np).trace()
        nb_sample += motion.shape[0]

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    r_precision_real = np.asarray(r_precision_real, dtype=np.float64) / nb_sample
    r_precision = np.asarray(r_precision, dtype=np.float64) / nb_sample
    metrics = {
        "fid": float(calculate_frechet_distance(gt_mu, gt_cov, mu, cov)),
        "diversity_real": float(diversity_real),
        "diversity": float(diversity),
        "top1_real": float(r_precision_real[0]),
        "top2_real": float(r_precision_real[1]),
        "top3_real": float(r_precision_real[2]),
        "top1": float(r_precision[0]),
        "top2": float(r_precision[1]),
        "top3": float(r_precision[2]),
        "matching_score_real": float(matching_score_real / nb_sample),
        "matching_score": float(matching_score_pred / nb_sample),
    }
    model.train()
    return metrics


def checkpoint_payload(args, model, optimizer, step: int, epoch: int, metrics=None, include_optimizer=False):
    payload = {
        "vq_model": model.state_dict(),
        "step": step,
        "ep": epoch,
        "config": vars(args),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def save_checkpoint(path: Path, args, model, optimizer, step: int, epoch: int, metrics=None, include_optimizer=False):
    payload = checkpoint_payload(args, model, optimizer, step, epoch, metrics, include_optimizer)
    torch.save(payload, path)


def refresh_best_aliases(model_dir: Path, key: str, ranking: list[dict]) -> None:
    for idx, entry in enumerate(ranking, start=1):
        shutil.copy2(entry["path"], model_dir / f"best_{key}_rank{idx}.tar")
    if ranking:
        alias = "net_best_fid.tar" if key == "fid" else "net_best_top3.tar"
        shutil.copy2(ranking[0]["path"], model_dir / alias)


def maybe_save_ranked(
    args,
    model,
    optimizer,
    model_dir: Path,
    rankings: dict[str, list[dict]],
    key: str,
    mode: str,
    value: float,
    step: int,
    epoch: int,
    metrics: dict[str, float],
) -> bool:
    current = rankings.setdefault(key, [])
    if len(current) >= args.topk:
        worst = current[-1]["value"]
        if mode == "min" and value >= worst:
            return False
        if mode == "max" and value <= worst:
            return False

    ckpt_dir = model_dir / f"best_{key}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"step_{step:09d}_{key}_{value:.6f}_fid_{metrics.get('fid', math.nan):.6f}_top3_{metrics.get('top3', math.nan):.6f}.tar"
    ckpt_path = ckpt_dir / ckpt_name
    save_checkpoint(ckpt_path, args, model, optimizer, step, epoch, metrics=metrics, include_optimizer=False)

    current.append({"step": step, "epoch": epoch, "value": value, "path": str(ckpt_path), "metrics": metrics})
    reverse = mode == "max"
    current.sort(key=lambda item: item["value"], reverse=reverse)
    del current[args.topk:]
    refresh_best_aliases(model_dir, key, current)
    return True


def write_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def load_eval_components(args: argparse.Namespace, device: torch.device):
    from models.t2m_eval_wrapper import EvaluatorModelWrapper
    from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
    from utils.get_opt import get_opt

    opt_path = Path("checkpoints") / args.dataset_name / "Comp_v6_KLD005" / "opt.txt"
    wrapper_opt = get_opt(str(opt_path), device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_loader, _ = get_dataset_motion_loader(str(opt_path), args.eval_batch_size, "val", device=device)
    return eval_loader, eval_wrapper


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if not torch.cuda.is_available() and args.gpu_id >= 0:
        raise RuntimeError("CUDA is not available in this process")
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    args.data_root = args.data_root.resolve()
    if args.kv_root is not None:
        args.kv_root = args.kv_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.partition_file = args.partition_file.resolve()
    paths = setup_paths(args)

    if not args.partition_file.exists():
        raise FileNotFoundError(args.partition_file)
    shutil.copy2(args.partition_file, paths["config"] / args.partition_file.name)
    write_json(paths["config"] / "options.json", vars(args))

    mean, std = load_stats(args, paths)
    train_dataset = FixedWindowMotionDataset(args.data_root, "train", args.window_size, mean, std)
    val_dataset = FixedWindowMotionDataset(args.data_root, "val", args.window_size, mean, std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay
    )
    param_count = sum(param.numel() for param in model.parameters())
    print(f"[model] params={param_count / 1_000_000:.3f}M device={device}")
    print(f"[stats] mean={mean.shape} std={std.shape} std_min={std.min():.6g} std_max={std.max():.6g}")
    print(f"[train] batches_per_epoch={len(train_loader)} total_iter={args.total_iter}")

    eval_loader = None
    eval_wrapper = None
    if not args.disable_eval and args.eval_iter > 0:
        eval_loader, eval_wrapper = load_eval_components(args, device)

    step = 0
    epoch = 0
    best_val = math.inf
    rankings = {"fid": [], "top3": []}
    latest_path = paths["model"] / "latest.tar"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["vq_model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("ep", 0))
        print(f"[resume] loaded {latest_path} step={step} epoch={epoch}")

    model.train()
    logs: OrderedDict[str, float] = OrderedDict()
    log_count = 0
    start_time = time.time()
    train_iter = iter(train_loader)
    while step < args.total_iter:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1
        lr = lr_at_step(args, step)
        set_optimizer_lr(optimizer, lr)

        motions = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        losses = compute_losses(args, model, motions)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()

        log_count += 1
        for key, value in losses.items():
            logs[key] = logs.get(key, 0.0) + float(value.detach().cpu().item())
        logs["lr"] = logs.get("lr", 0.0) + lr

        if step % args.print_iter == 0:
            avg = {key: value / log_count for key, value in logs.items()}
            elapsed = format_seconds(time.time() - start_time)
            msg = " ".join(f"{key}={value:.6f}" for key, value in avg.items())
            print(f"[train] step={step} epoch={epoch} elapsed={elapsed} {msg}", flush=True)
            logs.clear()
            log_count = 0

        if step % args.save_latest == 0 or step == args.total_iter:
            save_checkpoint(latest_path, args, model, optimizer, step, epoch, include_optimizer=True)

        if args.eval_iter > 0 and (step % args.eval_iter == 0 or step == args.total_iter):
            val_metrics = average_val_loss(args, model, val_loader, device)
            print(
                "[val] "
                + f"step={step} epoch={epoch} "
                + " ".join(f"{key}={value:.6f}" for key, value in val_metrics.items()),
                flush=True,
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    paths["model"] / "net_best_val.tar",
                    args,
                    model,
                    optimizer,
                    step,
                    epoch,
                    metrics={"val": val_metrics},
                    include_optimizer=False,
                )

            if eval_loader is not None and eval_wrapper is not None:
                eval_metrics = evaluate_retrieval(model, eval_loader, eval_wrapper, device)
                metrics = {"val": val_metrics, **eval_metrics}
                print(
                    "[eval] "
                    + f"step={step} epoch={epoch} "
                    + " ".join(f"{key}={value:.6f}" for key, value in eval_metrics.items()),
                    flush=True,
                )
                fid_hit = maybe_save_ranked(
                    args, model, optimizer, paths["model"], rankings, "fid", "min",
                    eval_metrics["fid"], step, epoch, metrics
                )
                top3_hit = maybe_save_ranked(
                    args, model, optimizer, paths["model"], rankings, "top3", "max",
                    eval_metrics["top3"], step, epoch, metrics
                )
                print(f"[best] step={step} fid_hit={fid_hit} top3_hit={top3_hit}", flush=True)
                write_json(paths["logs"] / "best_metrics.json", rankings)

    save_checkpoint(latest_path, args, model, optimizer, step, epoch, include_optimizer=True)
    print(f"[done] step={step} epoch={epoch} latest={latest_path}", flush=True)


if __name__ == "__main__":
    main()
