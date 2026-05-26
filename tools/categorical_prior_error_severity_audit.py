#!/usr/bin/env python3
"""Audit geometric severity of errors from a categorical motion-token prior.

The script is diagnostic-only. It loads the released KV-Control tokenizer and
base T-Concat v4 categorical prior, masks ground-truth motion tokens, and asks
whether wrong token predictions are geometrically near or far from the target
code in learned-code and motion-prototype spaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from motion_code_geometry_diagnostics import (
    VQ_CFG,
    codebooks_numpy,
    encode_dataset,
    ids_flat_to_grid,
    json_default,
    load_split_ids,
    load_vq_model,
    part_name,
    write_csv,
)


DEFAULT_KV_ROOT = Path(".")
DEFAULT_DATA_ROOT = Path("dataset/HumanML3D")
DEFAULT_OUT_DIR = Path("geometry_diagnostics/categorical_prior_audit")


@dataclass
class AuditBatch:
    names: List[str]
    captions: List[str]
    motions: np.ndarray
    lengths: np.ndarray


class GroupStats:
    def __init__(self) -> None:
        self.count = 0
        self.correct = 0
        self.top5 = 0
        self.nll_sum = 0.0
        self.pred_conf_sum = 0.0
        self.gt_prob_sum = 0.0
        self.expected_code_sum = 0.0
        self.expected_by_space_sum: Dict[str, float] = defaultdict(float)
        self.wrong_count = 0
        self.wrong_code_values: List[float] = []
        self.wrong_by_space_values: Dict[str, List[float]] = defaultdict(list)

    def add(
        self,
        correct: bool,
        top5: bool,
        nll: float,
        pred_conf: float,
        gt_prob: float,
        expected_code: float,
        expected_by_space: Dict[str, float],
        code_dist: float,
        damage_by_space: Dict[str, float],
    ) -> None:
        self.count += 1
        self.correct += int(correct)
        self.top5 += int(top5)
        self.nll_sum += nll
        self.pred_conf_sum += pred_conf
        self.gt_prob_sum += gt_prob
        self.expected_code_sum += expected_code
        for space, value in expected_by_space.items():
            if not math.isnan(value):
                self.expected_by_space_sum[space] += value
        if not correct:
            self.wrong_count += 1
            self.wrong_code_values.append(code_dist)
            for space, value in damage_by_space.items():
                if not math.isnan(value):
                    self.wrong_by_space_values[space].append(value)

    def to_row(self, key: Tuple[object, ...], spaces: Sequence[str]) -> Dict[str, object]:
        split, cond_scale, mask_ratio, part = key
        row: Dict[str, object] = {
            "split": split,
            "cond_scale": cond_scale,
            "mask_ratio": mask_ratio,
            "part": part,
            "part_name": "all" if int(part) < 0 else part_name(int(part)),
            "tokens": self.count,
            "accuracy": self.correct / max(1, self.count),
            "top5": self.top5 / max(1, self.count),
            "nll_mean": self.nll_sum / max(1, self.count),
            "pred_conf_mean": self.pred_conf_sum / max(1, self.count),
            "gt_prob_mean": self.gt_prob_sum / max(1, self.count),
            "wrong_tokens": self.wrong_count,
            "wrong_rate": self.wrong_count / max(1, self.count),
            "wrong_code_distance_mean": safe_mean(self.wrong_code_values),
            "wrong_code_distance_median": safe_median(self.wrong_code_values),
            "expected_code_distance_mean": self.expected_code_sum / max(1, self.count),
        }
        for space in spaces:
            vals = self.wrong_by_space_values.get(space, [])
            row[f"wrong_{space}_damage_mean"] = safe_mean(vals)
            row[f"wrong_{space}_damage_median"] = safe_median(vals)
            row[f"expected_{space}_damage_mean"] = self.expected_by_space_sum[space] / max(1, self.count)
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a teacher-forced geometric error-severity audit for KV-Control's categorical prior."
    )
    parser.add_argument("--kv-root", type=Path, default=DEFAULT_KV_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--prototype-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-audit-motions", type=int, default=512)
    parser.add_argument("--max-prototype-motions", type=int, default=512)
    parser.add_argument("--motion-length", type=int, default=196)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--mask-ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--cond-scales", type=float, nargs="+", default=[1.0, 3.25])
    parser.add_argument(
        "--prototype-spaces",
        nargs="+",
        default=["feature", "joint_pos"],
        choices=["feature", "joint_pos", "joint_pos_vel"],
    )
    parser.add_argument("--min-prototype-count", type=int, default=20)
    parser.add_argument("--clip-path", type=Path, default=None)
    parser.add_argument("--max-detail-rows", type=int, default=200000)
    parser.add_argument("--save-detail-rows", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use tiny defaults for a quick path check.")
    return parser.parse_args()


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def load_text_caption(text_path: Path) -> Optional[str]:
    if not text_path.is_file():
        return None
    fallback: Optional[str] = None
    with text_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("#")
            if len(parts) < 4:
                continue
            caption = parts[0].strip()
            if not caption:
                continue
            if fallback is None:
                fallback = caption
            try:
                start = float(parts[2])
                end = float(parts[3])
            except ValueError:
                continue
            if (np.isnan(start) or start == 0.0) and (np.isnan(end) or end == 0.0):
                return caption
    return fallback


def iter_audit_batches(
    data_root: Path,
    split: str,
    mean: np.ndarray,
    std: np.ndarray,
    motion_length: int,
    batch_size: int,
    max_motions: int,
    rng: np.random.Generator,
) -> Iterable[AuditBatch]:
    motion_dir = data_root / "new_joint_vecs"
    text_dir = data_root / "texts"
    ids = load_split_ids(data_root, split)

    names: List[str] = []
    captions: List[str] = []
    motions: List[np.ndarray] = []
    lengths: List[int] = []
    count = 0

    for name in ids:
        if max_motions > 0 and count >= max_motions:
            break
        motion_path = motion_dir / f"{name}.npy"
        if not motion_path.is_file():
            continue
        caption = load_text_caption(text_dir / f"{name}.txt")
        if caption is None:
            continue
        raw = np.load(motion_path)
        if raw.ndim != 2 or raw.shape[1] != mean.shape[0]:
            continue
        if len(raw) < 40:
            continue
        true_len = min(len(raw), motion_length)
        true_len = int((true_len // 4) * 4)
        if true_len < 4:
            continue
        raw = raw[:true_len].astype(np.float32, copy=False)
        if len(raw) < motion_length:
            pad = np.zeros((motion_length - len(raw), raw.shape[1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=0)
        norm = (raw - mean) / std

        names.append(name)
        captions.append(caption)
        motions.append(norm.astype(np.float32))
        lengths.append(true_len)
        count += 1

        if len(motions) == batch_size:
            yield AuditBatch(
                names=names,
                captions=captions,
                motions=np.stack(motions, axis=0),
                lengths=np.asarray(lengths, dtype=np.int32),
            )
            names, captions, motions, lengths = [], [], [], []

    if motions:
        yield AuditBatch(
            names=names,
            captions=captions,
            motions=np.stack(motions, axis=0),
            lengths=np.asarray(lengths, dtype=np.int32),
        )


def build_manual_transformer_opt() -> argparse.Namespace:
    return argparse.Namespace(
        name="base_t_concat_v4_manual",
        dataset_name="t2m",
        checkpoints_dir="./checkpoints",
        num_tokens=128,
        num_quantizers=6,
        code_dim=128,
        max_token_len=49,
        max_motion_length=196,
        unit_length=4,
        latent_dim=384,
        ff_size=1536,
        n_layers=20,
        n_heads=6,
        dropout=0.2,
        cond_drop_prob=0.1,
        transformer_variant="t_concat_v4",
        cross_attn_interval=2,
        cross_attn_heads=0,
        text_adapter_layers=4,
        gate_init=0.01,
        per_q_head=False,
        mask_2d_hybrid=False,
        mask_2d_ratio=0.3,
        factorized_attn="none",
        masked_only_ce=False,
        label_smoothing=0.0,
        noise_std=0.1,
        noise_masked_only=False,
    )


def load_base_transformer(kv_root: Path, device: torch.device):
    if str(kv_root.resolve()) not in sys.path:
        sys.path.insert(0, str(kv_root.resolve()))
    from kvctrl.models.mask_transformer.factory import build_mask_transformer

    opt = build_manual_transformer_opt()
    model = build_mask_transformer(
        "t_concat_v4",
        code_dim=opt.code_dim,
        cond_mode="text",
        latent_dim=opt.latent_dim,
        ff_size=opt.ff_size,
        num_layers=opt.n_layers,
        num_heads=opt.n_heads,
        dropout=opt.dropout,
        clip_dim=512,
        cond_drop_prob=opt.cond_drop_prob,
        clip_version="ViT-B/32",
        opt=opt,
    )
    ckpt_path = kv_root / "checkpoints" / "base_t_concat_v4" / "model" / "net_best_fid.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["t2m_transformer"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [key for key in missing if not key.startswith("clip_model.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"Transformer load mismatch. missing={bad_missing[:8]} unexpected={unexpected[:8]}")
    model.to(device).eval()
    return model, ckpt_path, int(ckpt.get("ep", -1))


def build_prototype_geometry(
    vq_model,
    kv_root: Path,
    data_root: Path,
    mean: np.ndarray,
    std: np.ndarray,
    partition: Dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], np.ndarray]:
    proto_args = argparse.Namespace(
        motion_length=args.motion_length,
        batch_size=args.batch_size,
        max_motions=args.max_prototype_motions,
        crop="start",
        prototype_spaces=args.prototype_spaces,
    )
    split_ids = load_split_ids(data_root, args.prototype_split)
    _encoded, proto_sums_by_space, counts, _meta = encode_dataset(
        vq_model,
        kv_root,
        data_root,
        mean,
        std,
        split_ids,
        partition,
        partition["partSeg"],
        proto_args,
        device,
        rng,
    )
    prototypes: Dict[str, List[np.ndarray]] = {}
    distance_mats: Dict[str, List[np.ndarray]] = {}
    for space, proto_sums in proto_sums_by_space.items():
        prototypes[space] = []
        distance_mats[space] = []
        for p, sums in enumerate(proto_sums):
            denom = counts[p, :, None].clip(min=1)
            proto = sums / denom
            valid = counts[p] >= args.min_prototype_count
            if np.any(valid):
                fill = proto[valid].mean(axis=0)
            else:
                fill = np.zeros((proto.shape[1],), dtype=np.float64)
            proto = proto.copy()
            proto[~valid] = fill
            diff = proto[:, None, :] - proto[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)
            prototypes[space].append(proto.astype(np.float32))
            distance_mats[space].append(dist)
    return prototypes, distance_mats, counts


def code_distance_mats(codebooks: Sequence[np.ndarray]) -> List[np.ndarray]:
    mats = []
    for emb in codebooks:
        diff = emb[:, None, :] - emb[None, :, :]
        mats.append(np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32))
    return mats


def make_frame_mask(
    lengths: np.ndarray,
    latent_len: int,
    mask_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros((len(lengths), latent_len), dtype=bool)
    for i, length in enumerate(lengths):
        valid_t = int(max(1, min(latent_len, length // 4)))
        n_mask = int(np.ceil(valid_t * mask_ratio))
        n_mask = max(1, min(valid_t, n_mask))
        chosen = rng.choice(valid_t, size=n_mask, replace=False)
        mask[i, chosen] = True
    return mask


def append_detail(
    detail_rows: List[Dict[str, object]],
    max_rows: int,
    row: Dict[str, object],
) -> None:
    if len(detail_rows) < max_rows:
        detail_rows.append(row)


def run_audit(
    transformer,
    vq_model,
    data_root: Path,
    mean: np.ndarray,
    std: np.ndarray,
    code_dists: Sequence[np.ndarray],
    damage_dists: Dict[str, List[np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    group_stats: Dict[Tuple[object, ...], GroupStats] = defaultdict(GroupStats)
    detail_rows: List[Dict[str, object]] = []
    confidence_bins: Dict[Tuple[object, ...], GroupStats] = defaultdict(GroupStats)

    with torch.no_grad():
        for batch_id, batch in enumerate(
            iter_audit_batches(
                data_root,
                args.split,
                mean,
                std,
                args.motion_length,
                args.batch_size,
                args.max_audit_motions,
                rng,
            )
        ):
            x = torch.from_numpy(batch.motions).to(device)
            ids_flat = vq_model(x, type="encode")
            ids_btq = ids_flat_to_grid(ids_flat.cpu(), 6).numpy().astype(np.int64)
            bsz, latent_len, num_parts = ids_btq.shape
            if num_parts != 6:
                raise RuntimeError(f"Expected 6 parts, got {num_parts}")

            valid_t = np.arange(latent_len)[None, :] < (batch.lengths[:, None] // 4)
            labels_np = ids_btq.reshape(bsz, latent_len * num_parts)
            non_pad_np = np.repeat(valid_t[:, :, None], num_parts, axis=2).reshape(bsz, latent_len * num_parts)
            padding_mask = torch.from_numpy(~non_pad_np).to(device)
            labels = torch.from_numpy(labels_np).long().to(device)

            cond_vector, text_seq, text_pad_mask = transformer.encode_text_with_seq(batch.captions)

            for cond_scale in args.cond_scales:
                for mask_ratio in args.mask_ratios:
                    frame_mask = make_frame_mask(batch.lengths, latent_len, mask_ratio, rng)
                    mask_np = np.repeat(frame_mask[:, :, None], num_parts, axis=2).reshape(
                        bsz, latent_len * num_parts
                    )
                    mask_np &= non_pad_np
                    input_ids = labels.clone()
                    input_ids[torch.from_numpy(mask_np).to(device)] = transformer.mask_id
                    input_ids[padding_mask] = transformer.pad_id
                    motion_emb = transformer.token_emb(input_ids)

                    logits = transformer.forward_with_cond_scale(
                        motion_emb,
                        cond_vector=cond_vector,
                        padding_mask=padding_mask,
                        cond_scale=cond_scale,
                        force_mask=False,
                        text_seq=text_seq,
                        text_pad_mask=text_pad_mask,
                    )
                    logits_bsv = logits.permute(0, 2, 1).contiguous()
                    probs = F.softmax(logits_bsv, dim=-1)
                    pred = logits_bsv.argmax(dim=-1)
                    top5 = torch.topk(logits_bsv, k=min(5, logits_bsv.shape[-1]), dim=-1).indices

                    masked_indices = np.argwhere(mask_np)
                    probs_cpu = probs.detach().cpu().numpy()
                    pred_cpu = pred.detach().cpu().numpy()
                    top5_cpu = top5.detach().cpu().numpy()

                    for bi, si in masked_indices:
                        part = int(si % num_parts)
                        latent_t = int(si // num_parts)
                        gt = int(labels_np[bi, si])
                        pred_id = int(pred_cpu[bi, si])
                        prob_vec = probs_cpu[bi, si]
                        gt_prob = float(max(prob_vec[gt], 1e-12))
                        pred_conf = float(prob_vec[pred_id])
                        is_correct = pred_id == gt
                        is_top5 = gt in top5_cpu[bi, si].tolist()
                        nll = float(-math.log(gt_prob))

                        code_dist = float(code_dists[part][gt, pred_id])
                        expected_code = float(np.dot(prob_vec, code_dists[part][gt]))
                        damage_by_space = {
                            space: float(mats[part][gt, pred_id])
                            for space, mats in damage_dists.items()
                        }
                        expected_by_space = {
                            space: float(np.dot(prob_vec, mats[part][gt]))
                            for space, mats in damage_dists.items()
                        }

                        keys = [
                            (args.split, cond_scale, mask_ratio, -1),
                            (args.split, cond_scale, mask_ratio, part),
                        ]
                        for key in keys:
                            group_stats[key].add(
                                is_correct,
                                is_top5,
                                nll,
                                pred_conf,
                                gt_prob,
                                expected_code,
                                expected_by_space,
                                code_dist,
                                damage_by_space,
                            )

                        conf_bin = min(4, int(pred_conf * 5.0))
                        confidence_bins[(args.split, cond_scale, mask_ratio, part, conf_bin)].add(
                            is_correct,
                            is_top5,
                            nll,
                            pred_conf,
                            gt_prob,
                            expected_code,
                            expected_by_space,
                            code_dist,
                            damage_by_space,
                        )

                        if args.save_detail_rows and (not is_correct):
                            row = {
                                "split": args.split,
                                "motion_name": batch.names[bi],
                                "caption": batch.captions[bi],
                                "cond_scale": cond_scale,
                                "mask_ratio": mask_ratio,
                                "latent_t": latent_t,
                                "part": part,
                                "part_name": part_name(part),
                                "gt_code": gt,
                                "pred_code": pred_id,
                                "pred_conf": pred_conf,
                                "gt_prob": gt_prob,
                                "nll": nll,
                                "code_distance": code_dist,
                                "expected_code_distance": expected_code,
                            }
                            for space, value in damage_by_space.items():
                                row[f"{space}_damage"] = value
                                row[f"expected_{space}_damage"] = expected_by_space[space]
                            append_detail(detail_rows, args.max_detail_rows, row)

            print(f"[audit] batch {batch_id + 1} complete, motions={sum(s.count for k, s in group_stats.items() if k[-1] == -1)}", flush=True)

    summary_rows = [
        stats.to_row(key, args.prototype_spaces)
        for key, stats in sorted(group_stats.items(), key=lambda item: item[0])
    ]
    conf_rows = []
    for key, stats in sorted(confidence_bins.items(), key=lambda item: item[0]):
        split, cond_scale, mask_ratio, part, conf_bin = key
        row = stats.to_row((split, cond_scale, mask_ratio, part), args.prototype_spaces)
        row["confidence_bin"] = conf_bin
        row["confidence_range"] = f"[{conf_bin / 5.0:.1f},{(conf_bin + 1) / 5.0:.1f})"
        conf_rows.append(row)
    return summary_rows, conf_rows, detail_rows


def write_report(
    out_dir: Path,
    summary: Dict[str, object],
    summary_rows: Sequence[Dict[str, object]],
    confidence_rows: Sequence[Dict[str, object]],
) -> None:
    rows_all = [r for r in summary_rows if int(r["part"]) == -1]
    lines = [
        "# Categorical Prior Error-Severity Audit",
        "",
        "## Inputs",
        f"- KV root: `{summary['kv_root']}`",
        f"- Data root: `{summary['data_root']}`",
        f"- Split: `{summary['split']}`",
        f"- Prototype split: `{summary['prototype_split']}`",
        f"- Audit motions: `{summary['max_audit_motions']}`",
        f"- Prototype motions: `{summary['max_prototype_motions']}`",
        f"- Base checkpoint: `{summary['base_checkpoint']}`",
        f"- VQ checkpoint: `{summary['vq_checkpoint']}`",
        f"- Checkpoint epoch: `{summary['checkpoint_epoch']}`",
        "",
        "## Overall Token Prediction",
        "| Cond scale | Mask ratio | Accuracy | Top-5 | Wrong code dist | Expected code dist |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_all:
        lines.append(
            f"| {float(r['cond_scale']):.2f} | {float(r['mask_ratio']):.2f} | "
            f"{float(r['accuracy']):.4f} | {float(r['top5']):.4f} | "
            f"{float(r['wrong_code_distance_mean']):.4f} | "
            f"{float(r['expected_code_distance_mean']):.4f} |"
        )

    for space in summary["prototype_spaces"]:
        lines.extend(
            [
                "",
                f"## {space} Damage",
                "| Cond scale | Mask ratio | Wrong top-1 damage | Expected damage |",
                "|---:|---:|---:|---:|",
            ]
        )
        for r in rows_all:
            lines.append(
                f"| {float(r['cond_scale']):.2f} | {float(r['mask_ratio']):.2f} | "
                f"{float(r[f'wrong_{space}_damage_mean']):.4f} | "
                f"{float(r[f'expected_{space}_damage_mean']):.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "This audit is diagnostic, not a fair flow comparison. It measures whether a",
            "real categorical text-to-token prior makes wrong predictions with non-uniform",
            "geometric severity, which CE treats as equal unit errors.",
            "",
            "## Files",
            "- `summary.json`",
            "- `audit_summary.csv`",
            "- `confidence_bins.csv`",
            "- `wrong_token_details.csv` when `--save-detail-rows` is enabled",
        ]
    )
    (out_dir / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_audit_motions = min(args.max_audit_motions, 8)
        args.max_prototype_motions = min(args.max_prototype_motions, 16)
        args.mask_ratios = [1.0]
        args.cond_scales = [1.0]
        args.prototype_spaces = ["feature"]
        args.batch_size = min(args.batch_size, 4)

    rng = seed_everything(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    clip_path = args.clip_path or (args.kv_root / "checkpoints" / "clip" / "ViT-B-32.pt")
    if not clip_path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint not found: {clip_path}")
    os.environ["MASKCONTROL_CLIP_MODEL_PATH"] = str(clip_path)

    mean = np.load(args.kv_root / "checkpoints" / "stats" / "mean.npy").astype(np.float32)
    std = np.load(args.kv_root / "checkpoints" / "stats" / "std.npy").astype(np.float32)
    std = np.where(std == 0, 1.0, std)

    vq_model, partition, vq_ckpt, partition_path = load_vq_model(args.kv_root, device)
    transformer, base_ckpt, checkpoint_epoch = load_base_transformer(args.kv_root, device)

    _prototypes, damage_dists, counts = build_prototype_geometry(
        vq_model,
        args.kv_root,
        args.data_root,
        mean,
        std,
        partition,
        args,
        device,
        rng,
    )
    code_dists = code_distance_mats(codebooks_numpy(vq_model))

    summary_rows, confidence_rows, detail_rows = run_audit(
        transformer,
        vq_model,
        args.data_root,
        mean,
        std,
        code_dists,
        damage_dists,
        args,
        device,
        rng,
    )

    summary = {
        "kv_root": str(args.kv_root),
        "data_root": str(args.data_root),
        "split": args.split,
        "prototype_split": args.prototype_split,
        "max_audit_motions": args.max_audit_motions,
        "max_prototype_motions": args.max_prototype_motions,
        "motion_length": args.motion_length,
        "mask_ratios": args.mask_ratios,
        "cond_scales": args.cond_scales,
        "prototype_spaces": args.prototype_spaces,
        "min_prototype_count": args.min_prototype_count,
        "base_checkpoint": str(base_ckpt),
        "vq_checkpoint": str(vq_ckpt),
        "partition_file": str(partition_path),
        "clip_path": str(clip_path),
        "checkpoint_epoch": checkpoint_epoch,
        "prototype_counts_min": counts.min(axis=1).tolist(),
        "prototype_counts_active": (counts >= args.min_prototype_count).sum(axis=1).tolist(),
    }

    write_csv(out_dir / "audit_summary.csv", summary_rows)
    write_csv(out_dir / "confidence_bins.csv", confidence_rows)
    if args.save_detail_rows:
        write_csv(out_dir / "wrong_token_details.csv", detail_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, summary, summary_rows, confidence_rows)
    print(f"[done] wrote audit to {out_dir}")


if __name__ == "__main__":
    main()
