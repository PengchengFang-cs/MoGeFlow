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
import torch.nn as nn
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
    parser.add_argument(
        "--mean_path",
        type=Path,
        default=None,
        help="Optional preprocessed normalization mean to use instead of data_root/Mean.npy.",
    )
    parser.add_argument(
        "--std_path",
        type=Path,
        default=None,
        help="Optional preprocessed normalization std to use instead of data_root/Std.npy. No feat_bias is applied when set.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--partition_file", type=Path, default=Path("configs/kit_skeleton_partition_formal_overlap.json"))

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
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
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--save_latest", type=int, default=500)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--disable_eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--part_specific_recipe",
        action="store_true",
        help="Use the part-aware-vqvae EMA reset and optimization schedule.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state(device: torch.device) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


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
    if args.mean_path is not None:
        mean = np.load(args.mean_path).astype(np.float32)
    else:
        mean = np.load(args.data_root / "Mean.npy").astype(np.float32)

    if args.std_path is not None:
        std = np.load(args.std_path).astype(np.float32)
    else:
        std_raw = np.load(args.data_root / "Std.npy").astype(np.float32)
        std = apply_feat_bias(std_raw, joints_num, args.feat_bias, args.stat_bias_passes).astype(np.float32)

    expected_dim = 4 + (joints_num - 1) * 9 + joints_num * 3 + 4
    if mean.shape[-1] != dim_pose or std.shape[-1] != dim_pose or expected_dim != dim_pose:
        raise ValueError(
            f"Unexpected feature dim: mean={mean.shape}, std={std.shape}, expected={dim_pose}"
        )

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
            self.data.append(motion)

        print(
            f"[dataset:{split}] motions={len(self.data)} sampling=uniform_motion_random_crop "
            f"missing={missing} short={short}"
        )
        if not self.data:
            raise RuntimeError(f"No motions loaded for split {split}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> torch.Tensor:
        source = self.data[item]
        idx = random.randint(0, len(source) - self.window_size)
        motion = source[idx:idx + self.window_size]
        motion = (motion - self.mean) / self.std
        return torch.from_numpy(motion.astype(np.float32))


class PartSpecificQuantizeEMAReset(nn.Module):
    """EMA reset quantizer from part-aware-vqvae commit 3c44777."""

    def __init__(self, nb_code: int, code_dim: int) -> None:
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer("codebook", torch.zeros(nb_code, code_dim))
        self.reset_count = 0
        self.usage = torch.zeros((nb_code, 1))

    def _tile(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < self.nb_code:
            repeats = (self.nb_code + x.shape[0] - 1) // x.shape[0]
            out = x.repeat(repeats, 1)
            return out + torch.randn_like(out) * (0.01 / np.sqrt(x.shape[1]))
        return x

    def init_codebook(self, x: torch.Tensor) -> None:
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).contiguous().view(-1, x.shape[1])

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        distance = (
            torch.sum(x ** 2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, self.codebook.t())
            + torch.sum(self.codebook ** 2, dim=1).unsqueeze(0)
        )
        return torch.min(distance, dim=-1).indices

    def dequantize(self, code_idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(code_idx, self.codebook)

    @torch.no_grad()
    def compute_perplexity(self, code_idx: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        onehot.scatter_(0, code_idx.view(1, -1), 1)
        prob = onehot.sum(dim=-1) / code_idx.shape[0]
        return torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

    @torch.no_grad()
    def update_codebook(self, x: torch.Tensor, code_idx: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        onehot.scatter_(0, code_idx.view(1, -1), 1)
        code_sum = torch.matmul(onehot, x)
        code_count = onehot.sum(dim=-1)
        out = self._tile(x)
        code_rand = out[torch.randperm(out.shape[0])[:self.nb_code]]
        self.code_sum = self.mu * self.code_sum + (1.0 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1.0 - self.mu) * code_count
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        self.usage = self.usage.to(usage.device)
        if self.reset_count >= 20:
            self.reset_count = 0
            usage = (usage + self.usage >= 1.0).float()
        else:
            self.reset_count += 1
            self.usage = (usage + self.usage >= 1.0).float()
            usage = torch.ones_like(self.usage, device=x.device)
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1.0 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        return torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_batch, _width, n_time = x.shape
        flat = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(flat)
        code_idx = self.quantize(flat)
        decoded = self.dequantize(code_idx)
        perplexity = self.update_codebook(flat, code_idx) if self.training else self.compute_perplexity(code_idx)
        commit_loss = F.mse_loss(flat, decoded.detach())
        decoded = flat + (decoded - flat).detach()
        decoded = decoded.view(n_batch, n_time, -1).permute(0, 2, 1).contiguous()
        return decoded, commit_loss, perplexity

    def get_code_idx(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, _width, n_time = x.shape
        flat = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(flat)
        return self.quantize(flat).view(n_batch, n_time).contiguous()

    def forward_from_code_idx(self, x_idx: torch.Tensor) -> torch.Tensor:
        n_batch, n_time = x_idx.shape
        decoded = self.dequantize(x_idx.long().view(-1))
        return decoded.view(n_batch, n_time, -1).permute(0, 2, 1).contiguous()


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
    )
    if args.part_specific_recipe:
        model.vqvae.quantizers = nn.ModuleList(
            [PartSpecificQuantizeEMAReset(args.nb_code, args.code_dim) for _ in model.vqvae.partSeg]
        )
    model = model.to(device)
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
        pred_pose_eval = torch.zeros_like(motion)
        for idx in range(motion.shape[0]):
            length = int(m_length[idx])
            pred_pose, _loss_commit, _perplexity = model(motion[idx:idx + 1, :length])
            pred_pose_eval[idx:idx + 1, :length] = pred_pose
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


def load_rankings(logs_dir: Path, model_dir: Path, topk: int) -> dict[str, list[dict]]:
    rankings = {"fid": [], "top3": []}
    metrics_path = logs_dir / "best_metrics.json"
    if not metrics_path.exists():
        return rankings

    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    for key, reverse in (("fid", False), ("top3", True)):
        entries = []
        for entry in payload.get(key, []):
            path = Path(entry.get("path", ""))
            if path.is_file():
                entries.append(entry)
        entries.sort(key=lambda item: item["value"], reverse=reverse)
        rankings[key] = entries[:topk]
        refresh_best_aliases(model_dir, key, rankings[key])
    return rankings


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
    eval_loader, _ = get_dataset_motion_loader(
        str(opt_path), args.eval_batch_size, args.eval_split, device=device
    )
    return eval_loader, eval_wrapper


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def main() -> None:
    args = parse_args()
    if args.eval_seed is None:
        args.eval_seed = args.seed
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
    if args.mean_path is not None:
        args.mean_path = args.mean_path.resolve()
    if args.std_path is not None:
        args.std_path = args.std_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.partition_file = args.partition_file.resolve()
    paths = setup_paths(args)

    if not args.partition_file.exists():
        raise FileNotFoundError(args.partition_file)
    shutil.copy2(args.partition_file, paths["config"] / args.partition_file.name)
    write_json(paths["config"] / "options.json", vars(args))

    mean, std = load_stats(args, paths)
    train_dataset = FixedWindowMotionDataset(args.data_root, "train", args.window_size, mean, std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
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
    rankings = {"fid": [], "top3": []}
    latest_path = paths["model"] / "latest.tar"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["vq_model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("ep", 0))
        print(f"[resume] loaded {latest_path} step={step} epoch={epoch}")
        rankings = load_rankings(paths["logs"], paths["model"], args.topk)
        print(
            f"[resume] fid_ranked={len(rankings['fid'])} top3_ranked={len(rankings['top3'])}"
        )

    model.train()

    def run_evaluation(eval_step: int) -> None:
        if eval_loader is None or eval_wrapper is None:
            return
        rng_state = capture_rng_state(device)
        try:
            set_seed(int(args.eval_seed))
            eval_metrics = evaluate_retrieval(model, eval_loader, eval_wrapper, device)
        finally:
            restore_rng_state(rng_state)
        metrics = dict(eval_metrics)
        print(
            "[eval] "
            + f"step={eval_step} epoch={epoch} "
            + " ".join(f"{key}={value:.6f}" for key, value in eval_metrics.items()),
            flush=True,
        )
        fid_hit = maybe_save_ranked(
            args, model, optimizer, paths["model"], rankings, "fid", "min",
            eval_metrics["fid"], eval_step, epoch, metrics
        )
        top3_hit = maybe_save_ranked(
            args, model, optimizer, paths["model"], rankings, "top3", "max",
            eval_metrics["top3"], eval_step, epoch, metrics
        )
        print(f"[best] step={eval_step} fid_hit={fid_hit} top3_hit={top3_hit}", flush=True)
        write_json(paths["logs"] / "best_metrics.json", rankings)

    logs: OrderedDict[str, float] = OrderedDict()
    log_count = 0
    start_time = time.time()
    train_iter = iter(train_loader)

    if args.part_specific_recipe and step == 0:
        for warmup_step in range(1, args.warm_up_iter):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_iter = iter(train_loader)
                batch = next(train_iter)
            lr = args.lr * float(warmup_step + 1) / float(args.warm_up_iter + 1)
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
            if warmup_step % args.print_iter == 0:
                avg = {key: value / log_count for key, value in logs.items()}
                msg = " ".join(f"{key}={value:.6f}" for key, value in avg.items())
                print(f"[warmup] step={warmup_step} epoch={epoch} {msg}", flush=True)
                logs.clear()
                log_count = 0
        logs.clear()
        log_count = 0
        run_evaluation(0)

    while step < args.total_iter:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1
        if args.part_specific_recipe:
            lr = float(optimizer.param_groups[0]["lr"])
        else:
            lr = lr_at_step(args, step)
            set_optimizer_lr(optimizer, lr)

        motions = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        losses = compute_losses(args, model, motions)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        if args.part_specific_recipe:
            scheduler.step()

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
            run_evaluation(step)

    save_checkpoint(latest_path, args, model, optimizer, step, epoch, include_optimizer=True)
    print(f"[done] step={step} epoch={epoch} latest={latest_path}", flush=True)


if __name__ == "__main__":
    main()
