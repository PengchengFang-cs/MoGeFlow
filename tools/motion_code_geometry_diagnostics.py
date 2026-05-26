#!/usr/bin/env python3
"""Geometry diagnostics for part-aware motion codebooks.

This script is intentionally read-only with respect to model checkpoints and
datasets. It loads the released KV-Control part-aware VQ tokenizer, encodes
HumanML3D motions, and measures whether learned codebook distances align with
local motion distances.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_KV_ROOT = Path(".")
DEFAULT_DATA_ROOT = Path("dataset/HumanML3D")
DEFAULT_OUT_DIR = Path("geometry_diagnostics")


VQ_CFG = {
    "dataname": "t2m",
    "batch_size": 256,
    "window_size": 64,
    "total_iter": 300000,
    "warm_up_iter": 1000,
    "lr": 2e-4,
    "lr_scheduler": [200000],
    "gamma": 0.05,
    "weight_decay": 0.0,
    "commit": 0.02,
    "loss_vel": 0.5,
    "recons_loss": "l1_smooth",
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
    "beta": 1.0,
    "resume_pth": None,
    "resume_gpt": None,
    "out_dir": "output",
    "results_dir": "visual_results/",
    "visual_name": "baseline",
    "exp_name": "exp_debug",
    "print_iter": 200,
    "eval_iter": 5000,
    "seed": 3407,
    "vis_gt": False,
    "nb_vis": 20,
    "sep_uplow": False,
}


PART_NAMES = [
    "root",
    "spine",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
]


@dataclass
class EncodedBatch:
    names: List[str]
    motions: np.ndarray
    lengths: np.ndarray
    latent_valid: np.ndarray
    ids: np.ndarray
    position_counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run geometry diagnostics for the KV-Control part-aware VQ codebook."
    )
    parser.add_argument("--kv-root", type=Path, default=DEFAULT_KV_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "train_val", "all"])
    parser.add_argument("--max-motions", type=int, default=512)
    parser.add_argument("--motion-length", type=int, default=196)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-code-count", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--nn-k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--replacement-samples", type=int, default=32)
    parser.add_argument("--crop", default="start", choices=["start", "center", "random"])
    parser.add_argument(
        "--prototype-spaces",
        nargs="+",
        default=["feature"],
        choices=["feature", "joint_pos", "joint_pos_vel"],
        help="Motion spaces used to build per-code prototypes.",
    )
    parser.add_argument("--confound-controls", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap samples for Spearman CI; 0 disables.")
    parser.add_argument("--error-cost", action="store_true", help="Write geometry-aware wrong-code cost tables.")
    parser.add_argument("--max-error-pairs", type=int, default=20000)
    parser.add_argument("--skip-replacement", action="store_true")
    parser.add_argument("--save-cache", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def add_kv_control_to_path(kv_root: Path) -> None:
    kv_root = kv_root.resolve()
    if str(kv_root) not in sys.path:
        sys.path.insert(0, str(kv_root))


def load_vq_model(kv_root: Path, device: torch.device):
    add_kv_control_to_path(kv_root)
    from argparse import Namespace
    from kvctrl.models import vqvae

    ckpt_path = kv_root / "checkpoints" / "vqvae" / "net_best_fid.pth"
    partition_path = kv_root / "checkpoints" / "vqvae" / "skeleton_partition.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"VQ checkpoint not found: {ckpt_path}")
    if not partition_path.is_file():
        raise FileNotFoundError(f"VQ partition not found: {partition_path}")

    cfg = dict(VQ_CFG)
    cfg["load_dir_vqvae"] = str(ckpt_path)
    cfg["partition_file"] = str(partition_path)
    args = Namespace(**cfg)

    model = vqvae.HumanVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"VQ load mismatch. missing={missing[:8]} unexpected={unexpected[:8]}")
    model.eval().to(device)

    with partition_path.open("r", encoding="utf-8") as f:
        partition = json.load(f)
    return model, partition, ckpt_path, partition_path


def load_split_ids(data_root: Path, split: str) -> List[str]:
    if split == "all":
        split_files = ["train.txt", "val.txt", "test.txt"]
        ids: List[str] = []
        for name in split_files:
            ids.extend(load_split_ids(data_root, name[:-4]))
        return sorted(set(ids))

    split_path = data_root / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def crop_or_pad_motion(
    motion: np.ndarray,
    motion_length: int,
    crop: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    true_len = min(len(motion), motion_length)
    if len(motion) > motion_length:
        if crop == "center":
            start = (len(motion) - motion_length) // 2
        elif crop == "random":
            start = int(rng.integers(0, len(motion) - motion_length + 1))
        else:
            start = 0
        motion = motion[start : start + motion_length]
        true_len = motion_length

    if len(motion) < motion_length:
        pad = np.zeros((motion_length - len(motion), motion.shape[1]), dtype=motion.dtype)
        motion = np.concatenate([motion, pad], axis=0)

    return motion.astype(np.float32, copy=False), int(true_len)


def iter_motion_batches(
    data_root: Path,
    ids: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    motion_length: int,
    batch_size: int,
    max_motions: int,
    crop: str,
    rng: np.random.Generator,
) -> Iterable[Tuple[List[str], np.ndarray, np.ndarray]]:
    motion_dir = data_root / "new_joint_vecs"
    names: List[str] = []
    motions: List[np.ndarray] = []
    lengths: List[int] = []

    count = 0
    for name in ids:
        if max_motions > 0 and count >= max_motions:
            break
        path = motion_dir / f"{name}.npy"
        if not path.is_file():
            continue
        raw = np.load(path)
        if raw.ndim != 2 or raw.shape[1] != mean.shape[0]:
            continue
        raw, valid_len = crop_or_pad_motion(raw, motion_length, crop, rng)
        norm = (raw - mean) / std
        names.append(name)
        motions.append(norm.astype(np.float32))
        lengths.append(valid_len)
        count += 1
        if len(motions) == batch_size:
            yield names, np.stack(motions, axis=0), np.asarray(lengths, dtype=np.int32)
            names, motions, lengths = [], [], []

    if motions:
        yield names, np.stack(motions, axis=0), np.asarray(lengths, dtype=np.int32)


def ids_flat_to_grid(ids_flat: torch.Tensor, num_parts: int) -> torch.Tensor:
    batch, total = ids_flat.shape
    if total % num_parts != 0:
        raise ValueError(f"Cannot reshape ids of length {total} into {num_parts} parts.")
    latent_len = total // num_parts
    return ids_flat.view(batch, num_parts, latent_len).permute(0, 2, 1).contiguous()


def latent_valid_mask(lengths: np.ndarray, latent_len: int, stride: int = 4) -> np.ndarray:
    starts = np.arange(latent_len) * stride
    ends = starts + stride
    return ends[None, :] <= lengths[:, None]


def patch_matrix(motions: np.ndarray, part_dims: Sequence[int], latent_len: int, stride: int = 4) -> np.ndarray:
    patches = []
    for t in range(latent_len):
        start = t * stride
        end = start + stride
        patches.append(motions[:, start:end, :][:, :, part_dims].reshape(motions.shape[0], -1))
    return np.stack(patches, axis=1).astype(np.float32)


def joint_patch_matrix(
    joints: np.ndarray,
    part_joints: Sequence[int],
    latent_len: int,
    include_velocity: bool,
    stride: int = 4,
) -> np.ndarray:
    if include_velocity:
        velocity = np.zeros_like(joints)
        velocity[:, 1:] = joints[:, 1:] - joints[:, :-1]
        source = np.concatenate([joints, velocity], axis=-1)
    else:
        source = joints

    patches = []
    for t in range(latent_len):
        start = t * stride
        end = start + stride
        patches.append(source[:, start:end, part_joints, :].reshape(joints.shape[0], -1))
    return np.stack(patches, axis=1).astype(np.float32)


def recover_joints_np(kv_root: Path, denorm_motion: np.ndarray, device: torch.device) -> np.ndarray:
    add_kv_control_to_path(kv_root)
    from kvctrl.utils.motion_process import recover_from_ric

    with torch.no_grad():
        x = torch.from_numpy(denorm_motion).float().to(device)
        joints = recover_from_ric(x, 22).detach().cpu().numpy()
    return joints.astype(np.float32)


def prototype_dim_for_space(space: str, part: Sequence[int], part_joints: Sequence[int]) -> int:
    if space == "feature":
        return 4 * len(part)
    if space == "joint_pos":
        return 4 * len(part_joints) * 3
    if space == "joint_pos_vel":
        return 4 * len(part_joints) * 6
    raise ValueError(f"Unknown prototype space: {space}")


def encode_dataset(
    model,
    kv_root: Path,
    data_root: Path,
    mean: np.ndarray,
    std: np.ndarray,
    ids: Sequence[str],
    partition: Dict[str, object],
    part_seg: Sequence[Sequence[int]],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[EncodedBatch, Dict[str, List[np.ndarray]], np.ndarray, Dict[str, object]]:
    all_names: List[str] = []
    all_motions: List[np.ndarray] = []
    all_lengths: List[np.ndarray] = []
    all_valid: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []

    num_parts = len(part_seg)
    part_joints = partition.get("part_joints_effective") or partition.get("part_joints_base")
    if part_joints is None:
        raise RuntimeError("Partition JSON does not contain part_joints_effective or part_joints_base.")
    part_joints = [list(map(int, joints)) for joints in part_joints]

    counts = np.zeros((num_parts, VQ_CFG["nb_code"]), dtype=np.int64)
    position_counts = np.zeros((num_parts, VQ_CFG["nb_code"], args.motion_length // 4), dtype=np.int64)
    proto_sums_by_space = {
        space: [
            np.zeros((VQ_CFG["nb_code"], prototype_dim_for_space(space, part_seg[p], part_joints[p])), dtype=np.float64)
            for p in range(num_parts)
        ]
        for space in args.prototype_spaces
    }

    total_motions = 0
    latent_len_seen: Optional[int] = None

    with torch.no_grad():
        for names, motions, lengths in iter_motion_batches(
            data_root,
            ids,
            mean,
            std,
            args.motion_length,
            args.batch_size,
            args.max_motions,
            args.crop,
            rng,
        ):
            x = torch.from_numpy(motions).to(device)
            ids_flat = model(x, type="encode")
            ids_grid = ids_flat_to_grid(ids_flat.cpu(), num_parts).numpy().astype(np.int16)
            latent_len = ids_grid.shape[1]
            if latent_len_seen is None:
                latent_len_seen = latent_len
            elif latent_len != latent_len_seen:
                raise RuntimeError(f"Mixed latent lengths: {latent_len_seen} and {latent_len}")

            valid = latent_valid_mask(lengths, latent_len)

            patch_by_space: Dict[str, List[np.ndarray]] = {}
            if "feature" in args.prototype_spaces:
                patch_by_space["feature"] = [
                    patch_matrix(motions, dims, latent_len)
                    for dims in part_seg
                ]
            if "joint_pos" in args.prototype_spaces or "joint_pos_vel" in args.prototype_spaces:
                denorm = motions * std[None, None, :] + mean[None, None, :]
                joints = recover_joints_np(kv_root, denorm.astype(np.float32), device)
                if "joint_pos" in args.prototype_spaces:
                    patch_by_space["joint_pos"] = [
                        joint_patch_matrix(joints, part_joints[p], latent_len, include_velocity=False)
                        for p in range(num_parts)
                    ]
                if "joint_pos_vel" in args.prototype_spaces:
                    patch_by_space["joint_pos_vel"] = [
                        joint_patch_matrix(joints, part_joints[p], latent_len, include_velocity=True)
                        for p in range(num_parts)
                    ]

            for p in range(num_parts):
                valid_ids = ids_grid[:, :, p][valid]
                if len(valid_ids) == 0:
                    continue
                counts[p] += np.bincount(valid_ids, minlength=VQ_CFG["nb_code"])
                for t in range(latent_len):
                    ids_t = ids_grid[:, t, p][valid[:, t]]
                    if len(ids_t):
                        position_counts[p, :, t] += np.bincount(ids_t, minlength=VQ_CFG["nb_code"])
                for space, part_patches in patch_by_space.items():
                    valid_patches = part_patches[p][valid]
                    np.add.at(proto_sums_by_space[space][p], valid_ids, valid_patches)

            all_names.extend(names)
            all_motions.append(motions)
            all_lengths.append(lengths)
            all_valid.append(valid)
            all_ids.append(ids_grid)
            total_motions += len(names)
            print(f"[encode] {total_motions} motions encoded", flush=True)

    if total_motions == 0:
        raise RuntimeError("No motions were encoded. Check data path and split.")

    encoded = EncodedBatch(
        names=all_names,
        motions=np.concatenate(all_motions, axis=0),
        lengths=np.concatenate(all_lengths, axis=0),
        latent_valid=np.concatenate(all_valid, axis=0),
        ids=np.concatenate(all_ids, axis=0),
        position_counts=position_counts,
    )
    meta = {
        "num_motions": total_motions,
        "motion_length": args.motion_length,
        "latent_length": int(encoded.ids.shape[1]),
        "num_parts": num_parts,
        "part_dims": [len(part) for part in part_seg],
        "prototype_spaces": list(args.prototype_spaces),
    }
    return encoded, proto_sums_by_space, counts, meta


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def pairwise_dist_rows(x: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    diff = x[pairs[:, 0]] - x[pairs[:, 1]]
    return np.sqrt(np.sum(diff * diff, axis=1))


def random_like_embedding(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random embedding with per-dimension mean/std matched to a reference codebook."""
    ref = reference.astype(np.float32)
    rand = rng.normal(size=ref.shape).astype(np.float32)
    rand = (rand - rand.mean(axis=0, keepdims=True)) / (rand.std(axis=0, keepdims=True) + 1e-6)
    return rand * ref.std(axis=0, keepdims=True) + ref.mean(axis=0, keepdims=True)


def sample_pairs(n: int, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    total = n * (n - 1) // 2
    if total <= max_pairs:
        rows, cols = np.triu_indices(n, k=1)
        return np.stack([rows, cols], axis=1)

    pairs = set()
    while len(pairs) < max_pairs:
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n - 1))
        if b >= a:
            b += 1
        if a > b:
            a, b = b, a
        pairs.add((a, b))
    return np.asarray(sorted(pairs), dtype=np.int64)


def nearest_indices(x: np.ndarray, active: np.ndarray, k: int) -> np.ndarray:
    x_active = x[active]
    dist = np.sum((x_active[:, None, :] - x_active[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(dist, np.inf)
    k_eff = min(k, len(active) - 1)
    return np.argsort(dist, axis=1)[:, :k_eff]


def neighborhood_overlap(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[1] == 0:
        return float("nan")
    vals = []
    for i in range(a.shape[0]):
        vals.append(len(set(a[i].tolist()).intersection(b[i].tolist())) / float(a.shape[1]))
    return float(np.mean(vals))


def compute_usage_rows(counts: np.ndarray) -> List[Dict[str, object]]:
    rows = []
    for p in range(counts.shape[0]):
        c = counts[p].astype(np.float64)
        total = float(c.sum())
        prob = c / total if total > 0 else c
        nonzero = c > 0
        perplexity = math.exp(float(-np.sum(prob[nonzero] * np.log(prob[nonzero] + 1e-12)))) if total > 0 else 0.0
        rows.append(
            {
                "part": p,
                "part_name": part_name(p),
                "total_tokens": int(total),
                "active_codes": int(np.sum(nonzero)),
                "dead_codes": int(np.sum(~nonzero)),
                "perplexity": perplexity,
                "min_count_active": int(c[nonzero].min()) if np.any(nonzero) else 0,
                "max_count": int(c.max()) if len(c) else 0,
            }
        )
    return rows


def part_name(index: int) -> str:
    if index < len(PART_NAMES):
        return PART_NAMES[index]
    return f"part_{index}"


def codebooks_numpy(model) -> List[np.ndarray]:
    return [q.codebook.detach().cpu().numpy().astype(np.float32) for q in model.vqvae.quantizers]


def compute_distance_and_nn(
    codebooks: List[np.ndarray],
    proto_sums_by_space: Dict[str, List[np.ndarray]],
    counts: np.ndarray,
    min_count: int,
    max_pairs: int,
    nn_ks: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    corr_rows: List[Dict[str, object]] = []
    nn_rows: List[Dict[str, object]] = []

    for space, proto_sums in proto_sums_by_space.items():
        for p, emb in enumerate(codebooks):
            active = np.flatnonzero(counts[p] >= min_count)
            if len(active) < 4:
                continue
            proto = proto_sums[p][active] / counts[p, active, None].clip(min=1)
            learned = emb[active]
            controls = {
                "learned": learned,
                "shuffled": learned[rng.permutation(len(learned))],
                "random_gaussian": random_like_embedding(learned, rng),
                "code_id_abs": active[:, None].astype(np.float32),
                "one_hot_hamming": np.eye(len(active), dtype=np.float32),
            }

            pairs = sample_pairs(len(active), max_pairs, rng)
            motion_d = pairwise_dist_rows(proto, pairs)
            for control_name, control_emb in controls.items():
                code_d = pairwise_dist_rows(control_emb, pairs)
                corr_rows.append(
                    {
                        "prototype_space": space,
                        "part": p,
                        "part_name": part_name(p),
                        "control": control_name,
                        "active_codes": len(active),
                        "pairs": len(pairs),
                        "spearman": spearman(code_d, motion_d),
                        "pearson": pearson(code_d, motion_d),
                        "code_dist_mean": float(np.mean(code_d)),
                        "motion_dist_mean": float(np.mean(motion_d)),
                    }
                )

            motion_nn_cache: Dict[int, np.ndarray] = {}
            for k in nn_ks:
                k_eff = min(k, len(active) - 1)
                if k_eff <= 0:
                    continue
                if k_eff not in motion_nn_cache:
                    motion_nn_cache[k_eff] = nearest_indices(proto, np.arange(len(active)), k_eff)
                motion_nn = motion_nn_cache[k_eff]
                random_motion_dist = []
                for i in range(len(active)):
                    candidates = [j for j in range(len(active)) if j != i]
                    sampled = rng.choice(candidates, size=k_eff, replace=False)
                    random_motion_dist.extend(np.linalg.norm(proto[i] - proto[sampled], axis=1).tolist())

                for control_name, control_emb in controls.items():
                    code_nn = nearest_indices(control_emb, np.arange(len(active)), k_eff)
                    nn_motion_d = []
                    for i in range(len(active)):
                        nn_motion_d.extend(np.linalg.norm(proto[i] - proto[code_nn[i]], axis=1).tolist())
                    nn_rows.append(
                        {
                            "prototype_space": space,
                            "part": p,
                            "part_name": part_name(p),
                            "control": control_name,
                            "k": k_eff,
                            "active_codes": len(active),
                            "nn_motion_dist_mean": float(np.mean(nn_motion_d)),
                            "random_motion_dist_mean": float(np.mean(random_motion_dist)),
                            "relative_improvement_vs_random": float(
                                (np.mean(random_motion_dist) - np.mean(nn_motion_d))
                                / (np.mean(random_motion_dist) + 1e-12)
                            ),
                            "overlap_with_motion_nn": neighborhood_overlap(code_nn, motion_nn),
                        }
                    )

    return corr_rows, nn_rows


def compute_trajectory_rows(
    encoded: EncodedBatch,
    codebooks: List[np.ndarray],
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    rows = []
    for p, emb in enumerate(codebooks):
        controls = {
            "learned": emb,
            "shuffled": emb[rng.permutation(len(emb))],
            "random_gaussian": random_like_embedding(emb, rng),
        }
        ids_p = encoded.ids[:, :, p].astype(np.int64)
        valid = encoded.latent_valid
        for control_name, control_emb in controls.items():
            steps = []
            curvs = []
            for n in range(ids_p.shape[0]):
                seq_ids = ids_p[n][valid[n]]
                if len(seq_ids) < 2:
                    continue
                z = control_emb[seq_ids]
                dz = z[1:] - z[:-1]
                steps.extend(np.linalg.norm(dz, axis=1).tolist())
                if len(z) >= 3:
                    ddz = z[2:] - 2.0 * z[1:-1] + z[:-2]
                    curvs.extend(np.linalg.norm(ddz, axis=1).tolist())
            rows.append(
                {
                    "part": p,
                    "part_name": part_name(p),
                    "control": control_name,
                    "num_steps": len(steps),
                    "step_mean": float(np.mean(steps)) if steps else float("nan"),
                    "step_median": float(np.median(steps)) if steps else float("nan"),
                    "curvature_mean": float(np.mean(curvs)) if curvs else float("nan"),
                    "curvature_median": float(np.median(curvs)) if curvs else float("nan"),
                }
            )
    return rows


def sample_pairs_matching_bins(
    bins_a: np.ndarray,
    bins_b: Optional[np.ndarray],
    max_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups: Dict[Tuple[int, int], List[int]] = {}
    if bins_b is None:
        bins_b = np.zeros_like(bins_a)
    for idx, key in enumerate(zip(bins_a.tolist(), bins_b.tolist())):
        groups.setdefault((int(key[0]), int(key[1])), []).append(idx)

    candidates = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members_arr = np.asarray(members, dtype=np.int64)
        rows, cols = np.triu_indices(len(members_arr), k=1)
        group_pairs = np.stack([members_arr[rows], members_arr[cols]], axis=1)
        candidates.append(group_pairs)
    if not candidates:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.concatenate(candidates, axis=0)
    if len(pairs) > max_pairs:
        keep = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = pairs[keep]
    return pairs.astype(np.int64)


def code_frequency_bins(counts: np.ndarray, active: np.ndarray, n_bins: int = 4) -> np.ndarray:
    vals = counts[active].astype(np.float64)
    order = np.argsort(vals, kind="mergesort")
    bins = np.zeros(len(active), dtype=np.int64)
    for rank, idx in enumerate(order):
        bins[idx] = min(n_bins - 1, int(rank * n_bins / max(1, len(active))))
    return bins


def code_time_bins(position_counts: np.ndarray, active: np.ndarray, latent_len: int, n_bins: int = 4) -> np.ndarray:
    peaks = np.argmax(position_counts[active], axis=1)
    bins = np.floor(peaks * n_bins / max(1, latent_len)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)


def compute_stratified_correlations(
    codebooks: List[np.ndarray],
    proto_sums_by_space: Dict[str, List[np.ndarray]],
    counts: np.ndarray,
    position_counts: np.ndarray,
    min_count: int,
    max_pairs: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    latent_len = position_counts.shape[-1]
    for space, proto_sums in proto_sums_by_space.items():
        for p, emb in enumerate(codebooks):
            active = np.flatnonzero(counts[p] >= min_count)
            if len(active) < 4:
                continue
            proto = proto_sums[p][active] / counts[p, active, None].clip(min=1)
            learned = emb[active]
            shuffled = learned[rng.permutation(len(learned))]
            freq_bins = code_frequency_bins(counts[p], active)
            time_bins = code_time_bins(position_counts[p], active, latent_len)
            pair_sets = {
                "all": sample_pairs(len(active), max_pairs, rng),
                "frequency_matched": sample_pairs_matching_bins(freq_bins, None, max_pairs, rng),
                "time_matched": sample_pairs_matching_bins(time_bins, None, max_pairs, rng),
                "frequency_and_time_matched": sample_pairs_matching_bins(freq_bins, time_bins, max_pairs, rng),
            }
            for pair_name, pairs in pair_sets.items():
                if len(pairs) < 4:
                    continue
                motion_d = pairwise_dist_rows(proto, pairs)
                for control_name, control_emb in {"learned": learned, "shuffled": shuffled}.items():
                    code_d = pairwise_dist_rows(control_emb, pairs)
                    rows.append(
                        {
                            "prototype_space": space,
                            "part": p,
                            "part_name": part_name(p),
                            "pair_control": pair_name,
                            "embedding_control": control_name,
                            "active_codes": len(active),
                            "pairs": len(pairs),
                            "spearman": spearman(code_d, motion_d),
                            "pearson": pearson(code_d, motion_d),
                        }
                    )
    return rows


def bootstrap_rank_ci(
    code_d: np.ndarray,
    motion_d: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    if n_boot <= 0 or len(code_d) < 4:
        value = spearman(code_d, motion_d)
        return value, float("nan"), float("nan")
    x = rankdata(code_d)
    y = rankdata(motion_d)
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        vals[i] = pearson(x[idx], y[idx])
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compute_bootstrap_rows(
    codebooks: List[np.ndarray],
    proto_sums_by_space: Dict[str, List[np.ndarray]],
    counts: np.ndarray,
    min_count: int,
    max_pairs: int,
    n_boot: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if n_boot <= 0:
        return rows
    for space, proto_sums in proto_sums_by_space.items():
        for p, emb in enumerate(codebooks):
            active = np.flatnonzero(counts[p] >= min_count)
            if len(active) < 4:
                continue
            proto = proto_sums[p][active] / counts[p, active, None].clip(min=1)
            learned = emb[active]
            controls = {
                "learned": learned,
                "shuffled": learned[rng.permutation(len(learned))],
            }
            pairs = sample_pairs(len(active), max_pairs, rng)
            motion_d = pairwise_dist_rows(proto, pairs)
            for control_name, control_emb in controls.items():
                code_d = pairwise_dist_rows(control_emb, pairs)
                mean, low, high = bootstrap_rank_ci(code_d, motion_d, n_boot, rng)
                rows.append(
                    {
                        "prototype_space": space,
                        "part": p,
                        "part_name": part_name(p),
                        "control": control_name,
                        "pairs": len(pairs),
                        "bootstrap": n_boot,
                        "spearman_boot_mean": mean,
                        "spearman_ci_low": low,
                        "spearman_ci_high": high,
                    }
                )
    return rows


def compute_error_cost_rows(
    codebooks: List[np.ndarray],
    proto_sums_by_space: Dict[str, List[np.ndarray]],
    counts: np.ndarray,
    min_count: int,
    max_pairs: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for space, proto_sums in proto_sums_by_space.items():
        for p, emb in enumerate(codebooks):
            active = np.flatnonzero(counts[p] >= min_count)
            if len(active) < 4:
                continue
            proto = proto_sums[p][active] / counts[p, active, None].clip(min=1)
            learned = emb[active]
            pairs = sample_pairs(len(active), max_pairs, rng)
            code_d = pairwise_dist_rows(learned, pairs)
            motion_d = pairwise_dist_rows(proto, pairs)
            quantiles = np.quantile(code_d, [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(code_d, quantiles).astype(np.int64)
            for i, pair in enumerate(pairs):
                rows.append(
                    {
                        "prototype_space": space,
                        "part": p,
                        "part_name": part_name(p),
                        "code_i": int(active[pair[0]]),
                        "code_j": int(active[pair[1]]),
                        "ce_wrong_code_penalty": 1.0,
                        "code_distance": float(code_d[i]),
                        "motion_damage": float(motion_d[i]),
                        "code_distance_bin": int(bins[i]),
                    }
                )
    return rows


def summarize_error_cost(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    result = []
    groups = sorted(set((r["prototype_space"], r["part"], r["code_distance_bin"]) for r in rows))
    for space, part, bin_id in groups:
        subset = [r for r in rows if r["prototype_space"] == space and r["part"] == part and r["code_distance_bin"] == bin_id]
        result.append(
            {
                "prototype_space": space,
                "part": part,
                "part_name": part_name(int(part)),
                "code_distance_bin": bin_id,
                "pairs": len(subset),
                "code_distance_mean": float(np.mean([float(r["code_distance"]) for r in subset])),
                "motion_damage_mean": float(np.mean([float(r["motion_damage"]) for r in subset])),
                "motion_damage_median": float(np.median([float(r["motion_damage"]) for r in subset])),
            }
        )
    return result


def choose_near_mid_far(codebook: np.ndarray, code_id: int) -> Dict[str, int]:
    diff = codebook - codebook[code_id]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    order = np.argsort(dist)
    near = int(order[1]) if len(order) > 1 else int(code_id)
    mid = int(order[len(order) // 2])
    far = int(order[-1])
    return {"near": near, "mid": mid, "far": far}


def compute_replacement_rows(
    model,
    encoded: EncodedBatch,
    codebooks: List[np.ndarray],
    part_seg: Sequence[Sequence[int]],
    replacement_samples: int,
    device: torch.device,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if replacement_samples <= 0:
        return rows

    n_total = encoded.ids.shape[0]
    sample_indices = rng.choice(n_total, size=min(replacement_samples, n_total), replace=False)

    with torch.no_grad():
        for sample_rank, n in enumerate(sample_indices):
            ids = encoded.ids[n : n + 1].astype(np.int64)
            valid_ts = np.flatnonzero(encoded.latent_valid[n])
            if len(valid_ts) == 0:
                continue
            original_ids_t = torch.from_numpy(ids).long().to(device)
            original_rec = model.vqvae.forward_decoder(original_ids_t).detach().cpu().numpy()[0]
            valid_frames = np.arange(encoded.motions.shape[1]) < encoded.lengths[n]
            for p in range(ids.shape[-1]):
                t = int(rng.choice(valid_ts))
                old_code = int(ids[0, t, p])
                repl = choose_near_mid_far(codebooks[p], old_code)
                repl["random"] = int(rng.choice([i for i in range(len(codebooks[p])) if i != old_code]))
                frame_start = t * 4
                frame_end = min(frame_start + 4, int(encoded.lengths[n]))
                local_frames = np.arange(frame_start, frame_end)
                for category, new_code in repl.items():
                    mod_ids = ids.copy()
                    mod_ids[0, t, p] = new_code
                    mod_rec = model.vqvae.forward_decoder(torch.from_numpy(mod_ids).long().to(device))
                    mod_np = mod_rec.detach().cpu().numpy()[0]
                    delta = np.abs(mod_np - original_rec)
                    whole = float(delta[valid_frames].mean()) if np.any(valid_frames) else float("nan")
                    target = float(delta[np.ix_(valid_frames, part_seg[p])].mean()) if np.any(valid_frames) else float("nan")
                    if len(local_frames) > 0:
                        local_target = float(delta[np.ix_(local_frames, part_seg[p])].mean())
                    else:
                        local_target = float("nan")
                    rows.append(
                        {
                            "sample_rank": sample_rank,
                            "motion_name": encoded.names[n],
                            "part": p,
                            "part_name": part_name(p),
                            "latent_t": t,
                            "old_code": old_code,
                            "replacement_type": category,
                            "new_code": new_code,
                            "code_distance": float(np.linalg.norm(codebooks[p][old_code] - codebooks[p][new_code])),
                            "whole_motion_delta_l1": whole,
                            "target_part_delta_l1": target,
                            "local_target_delta_l1": local_target,
                        }
                    )
            print(f"[replacement] {sample_rank + 1}/{len(sample_indices)} samples", flush=True)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_report(
    out_dir: Path,
    summary: Dict[str, object],
    usage_rows: Sequence[Dict[str, object]],
    corr_rows: Sequence[Dict[str, object]],
    nn_rows: Sequence[Dict[str, object]],
    trajectory_rows: Sequence[Dict[str, object]],
    replacement_rows: Sequence[Dict[str, object]],
    stratified_rows: Sequence[Dict[str, object]],
    bootstrap_rows: Sequence[Dict[str, object]],
    error_cost_summary_rows: Sequence[Dict[str, object]],
    replacement_summary_rows: Sequence[Dict[str, object]],
) -> None:
    learned_corr = [r for r in corr_rows if r["control"] == "learned"]
    shuffled_corr = [r for r in corr_rows if r["control"] == "shuffled"]
    replacement_summary = summarize_replacement(replacement_rows)
    lines = [
        "# Motion Code Geometry Diagnostics",
        "",
        "## Inputs",
        f"- KV root: `{summary['kv_root']}`",
        f"- Data root: `{summary['data_root']}`",
        f"- VQ checkpoint: `{summary['vq_checkpoint']}`",
        f"- Partition file: `{summary['partition_file']}`",
        f"- Split: `{summary['split']}`",
        f"- Motions encoded: `{summary['num_motions']}`",
        f"- Latent length: `{summary['latent_length']}`",
        "",
        "## Code Usage",
        "| Part | Active | Dead | Perplexity | Tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in usage_rows:
        lines.append(
            f"| {r['part_name']} | {r['active_codes']} | {r['dead_codes']} | "
            f"{float(r['perplexity']):.3f} | {r['total_tokens']} |"
        )
    lines.extend(["", "## Distance Correlation", "| Space | Part | Learned Spearman | Shuffled Spearman |", "|---|---|---:|---:|"])
    for r in learned_corr:
        s = next((x for x in shuffled_corr if x["part"] == r["part"] and x.get("prototype_space") == r.get("prototype_space")), None)
        s_val = float(s["spearman"]) if s is not None else float("nan")
        lines.append(f"| {r.get('prototype_space', 'feature')} | {r['part_name']} | {float(r['spearman']):.4f} | {s_val:.4f} |")

    lines.extend(["", "## Nearest-Neighbor Retrieval", "| Space | Part | k | Learned improvement vs random | Motion-NN overlap |", "|---|---|---:|---:|---:|"])
    for r in nn_rows:
        if r["control"] == "learned":
            lines.append(
                f"| {r.get('prototype_space', 'feature')} | {r['part_name']} | {r['k']} | "
                f"{float(r['relative_improvement_vs_random']):.4f} | "
                f"{float(r['overlap_with_motion_nn']):.4f} |"
            )

    if stratified_rows:
        lines.extend(
            [
                "",
                "## Stratified Correlation Controls",
                "| Space | Part | Pair control | Learned Spearman | Shuffled Spearman | Pairs |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for r in stratified_rows:
            if r["embedding_control"] != "learned":
                continue
            s = next(
                (
                    x
                    for x in stratified_rows
                    if x["prototype_space"] == r["prototype_space"]
                    and x["part"] == r["part"]
                    and x["pair_control"] == r["pair_control"]
                    and x["embedding_control"] == "shuffled"
                ),
                None,
            )
            s_val = float(s["spearman"]) if s is not None else float("nan")
            lines.append(
                f"| {r['prototype_space']} | {r['part_name']} | {r['pair_control']} | "
                f"{float(r['spearman']):.4f} | {s_val:.4f} | {r['pairs']} |"
            )

    if bootstrap_rows:
        lines.extend(
            [
                "",
                "## Bootstrap CI",
                "| Space | Part | Control | Spearman mean | 95% CI |",
                "|---|---|---|---:|---:|",
            ]
        )
        for r in bootstrap_rows:
            if r["control"] in {"learned", "shuffled"}:
                lines.append(
                    f"| {r['prototype_space']} | {r['part_name']} | {r['control']} | "
                    f"{float(r['spearman_boot_mean']):.4f} | "
                    f"[{float(r['spearman_ci_low']):.4f}, {float(r['spearman_ci_high']):.4f}] |"
                )

    lines.extend(["", "## Trajectory Smoothness", "| Part | Control | Step mean | Curvature mean |", "|---|---|---:|---:|"])
    for r in trajectory_rows:
        lines.append(
            f"| {r['part_name']} | {r['control']} | "
            f"{float(r['step_mean']):.4f} | {float(r['curvature_mean']):.4f} |"
        )

    if replacement_summary:
        lines.extend(["", "## Replacement Summary", "| Replacement | Whole delta | Target delta | Local target delta |", "|---|---:|---:|---:|"])
        for name in ["near", "mid", "far", "random"]:
            if name not in replacement_summary:
                continue
            r = replacement_summary[name]
            lines.append(
                f"| {name} | {r['whole_motion_delta_l1']:.5f} | "
                f"{r['target_part_delta_l1']:.5f} | {r['local_target_delta_l1']:.5f} |"
            )

    if replacement_summary_rows:
        lines.extend(
            [
                "",
                "## Replacement Per-Part Summary",
                "| Part | Replacement | Local target delta | 95% CI | Count |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for r in replacement_summary_rows:
            lines.append(
                f"| {r['part_name']} | {r['replacement_type']} | "
                f"{float(r['local_target_delta_l1_mean']):.5f} | "
                f"[{float(r['local_target_delta_l1_ci_low']):.5f}, {float(r['local_target_delta_l1_ci_high']):.5f}] | "
                f"{r['count']} |"
            )

    if error_cost_summary_rows:
        lines.extend(
            [
                "",
                "## Geometry-Aware Error Cost",
                "| Space | Part | Code-distance bin | Motion damage mean | Pairs |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for r in error_cost_summary_rows:
            lines.append(
                f"| {r['prototype_space']} | {r['part_name']} | {r['code_distance_bin']} | "
                f"{float(r['motion_damage_mean']):.5f} | {r['pairs']} |"
            )

    lines.extend(
        [
            "",
            "## Files",
            "- `code_usage.csv`",
            "- `distance_correlation.csv`",
            "- `nn_retrieval.csv`",
            "- `distance_correlation_stratified.csv`",
            "- `bootstrap_ci.csv`",
            "- `geometry_error_cost.csv`",
            "- `geometry_error_cost_summary.csv`",
            "- `trajectory_smoothness.csv`",
            "- `replacement_near_mid_far.csv`",
            "- `replacement_error_cost.csv`",
            "- `summary.json`",
        ]
    )
    (out_dir / "GEOMETRY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_replacement(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    if not rows:
        return result
    for category in sorted(set(r["replacement_type"] for r in rows)):
        subset = [r for r in rows if r["replacement_type"] == category]
        result[category] = {
            "whole_motion_delta_l1": float(np.mean([float(r["whole_motion_delta_l1"]) for r in subset])),
            "target_part_delta_l1": float(np.mean([float(r["target_part_delta_l1"]) for r in subset])),
            "local_target_delta_l1": float(np.mean([float(r["local_target_delta_l1"]) for r in subset])),
        }
    return result


def mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, mean, mean
    half = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return mean, mean - half, mean + half


def summarize_replacement_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    if not rows:
        return result
    groups = sorted(set((int(r["part"]), str(r["replacement_type"])) for r in rows))
    for part, repl in groups:
        subset = [r for r in rows if int(r["part"]) == part and str(r["replacement_type"]) == repl]
        whole = mean_ci([float(r["whole_motion_delta_l1"]) for r in subset])
        target = mean_ci([float(r["target_part_delta_l1"]) for r in subset])
        local = mean_ci([float(r["local_target_delta_l1"]) for r in subset])
        result.append(
            {
                "part": part,
                "part_name": part_name(part),
                "replacement_type": repl,
                "count": len(subset),
                "whole_motion_delta_l1_mean": whole[0],
                "whole_motion_delta_l1_ci_low": whole[1],
                "whole_motion_delta_l1_ci_high": whole[2],
                "target_part_delta_l1_mean": target[0],
                "target_part_delta_l1_ci_low": target[1],
                "target_part_delta_l1_ci_high": target[2],
                "local_target_delta_l1_mean": local[0],
                "local_target_delta_l1_ci_low": local[1],
                "local_target_delta_l1_ci_high": local[2],
            }
        )
    return result


def main() -> None:
    args = parse_args()
    rng = seed_everything(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, partition, ckpt_path, partition_path = load_vq_model(args.kv_root, device)
    part_seg = partition["partSeg"]
    mean = np.load(args.kv_root / "checkpoints" / "stats" / "mean.npy").astype(np.float32)
    std = np.load(args.kv_root / "checkpoints" / "stats" / "std.npy").astype(np.float32)
    std = np.where(std == 0, 1.0, std)

    split_ids = load_split_ids(args.data_root, args.split)
    encoded, proto_sums_by_space, counts, meta = encode_dataset(
        model,
        args.kv_root,
        args.data_root,
        mean,
        std,
        split_ids,
        partition,
        part_seg,
        args,
        device,
        rng,
    )
    codebooks = codebooks_numpy(model)

    usage_rows = compute_usage_rows(counts)
    corr_rows, nn_rows = compute_distance_and_nn(
        codebooks,
        proto_sums_by_space,
        counts,
        args.min_code_count,
        args.max_pairs,
        args.nn_k,
        rng,
    )
    stratified_rows: List[Dict[str, object]] = []
    if args.confound_controls:
        stratified_rows = compute_stratified_correlations(
            codebooks,
            proto_sums_by_space,
            counts,
            encoded.position_counts,
            args.min_code_count,
            args.max_pairs,
            rng,
        )
    bootstrap_rows = compute_bootstrap_rows(
        codebooks,
        proto_sums_by_space,
        counts,
        args.min_code_count,
        args.max_pairs,
        args.bootstrap,
        rng,
    )
    error_cost_rows: List[Dict[str, object]] = []
    error_cost_summary_rows: List[Dict[str, object]] = []
    if args.error_cost:
        error_cost_rows = compute_error_cost_rows(
            codebooks,
            proto_sums_by_space,
            counts,
            args.min_code_count,
            args.max_error_pairs,
            rng,
        )
        error_cost_summary_rows = summarize_error_cost(error_cost_rows)
    trajectory_rows = compute_trajectory_rows(encoded, codebooks, rng)
    replacement_rows: List[Dict[str, object]] = []
    if not args.skip_replacement:
        replacement_rows = compute_replacement_rows(
            model,
            encoded,
            codebooks,
            part_seg,
            args.replacement_samples,
            device,
            rng,
        )
    replacement_summary_rows = summarize_replacement_rows(replacement_rows)

    summary = {
        **meta,
        "kv_root": str(args.kv_root),
        "data_root": str(args.data_root),
        "vq_checkpoint": str(ckpt_path),
        "partition_file": str(partition_path),
        "split": args.split,
        "max_motions": args.max_motions,
        "min_code_count": args.min_code_count,
        "max_pairs": args.max_pairs,
        "bootstrap": args.bootstrap,
        "confound_controls": bool(args.confound_controls),
        "error_cost": bool(args.error_cost),
        "replacement_samples": args.replacement_samples,
        "code_usage": usage_rows,
        "replacement_summary": summarize_replacement(replacement_rows),
    }

    write_csv(out_dir / "code_usage.csv", usage_rows)
    write_csv(out_dir / "distance_correlation.csv", corr_rows)
    write_csv(out_dir / "nn_retrieval.csv", nn_rows)
    write_csv(out_dir / "distance_correlation_stratified.csv", stratified_rows)
    write_csv(out_dir / "bootstrap_ci.csv", bootstrap_rows)
    write_csv(out_dir / "geometry_error_cost.csv", error_cost_rows)
    write_csv(out_dir / "geometry_error_cost_summary.csv", error_cost_summary_rows)
    write_csv(out_dir / "trajectory_smoothness.csv", trajectory_rows)
    write_csv(out_dir / "replacement_near_mid_far.csv", replacement_rows)
    write_csv(out_dir / "replacement_error_cost.csv", replacement_summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8")

    if args.save_cache:
        np.savez_compressed(
            out_dir / "encoded_cache.npz",
            names=np.asarray(encoded.names),
            motions=encoded.motions,
            lengths=encoded.lengths,
            latent_valid=encoded.latent_valid,
            ids=encoded.ids,
            counts=counts,
            position_counts=encoded.position_counts,
        )

    write_report(
        out_dir,
        summary,
        usage_rows,
        corr_rows,
        nn_rows,
        trajectory_rows,
        replacement_rows,
        stratified_rows,
        bootstrap_rows,
        error_cost_summary_rows,
        replacement_summary_rows,
    )
    print(f"[done] wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
