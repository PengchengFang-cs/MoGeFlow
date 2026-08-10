#!/usr/bin/env python3
"""Formal geometry diagnostics for overlap-partition motion codebooks.

This is a separate entry point from ``motion_code_geometry_diagnostics.py``.
It keeps the original script untouched while fixing the formal-run assumptions:
checkpoint, partition, normalization stats, part names, and joint count are all
resolved from explicit inputs instead of the legacy hard-coded VQ bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import motion_code_geometry_diagnostics as diag


DEFAULT_KV_ROOT = Path(".")
DEFAULT_DATA_ROOT = Path("dataset/HumanML3D")
DEFAULT_OUT_DIR = Path("geometry_diagnostics/formal_hml3d")
FORMAL_PART_NAMES = [
    "root_lower_spine",
    "upper_torso_arms",
    "right_leg",
    "upper_spine_neck",
    "left_leg",
    "head",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal geometry diagnostics for the overlap-partition VQ codebook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kv-root", type=Path, default=DEFAULT_KV_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vq-checkpoint", type=Path, default=None)
    parser.add_argument("--partition-file", type=Path, default=None)
    parser.add_argument("--mean-path", type=Path, default=None)
    parser.add_argument("--std-path", type=Path, default=None)
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
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--error-cost", action="store_true")
    parser.add_argument("--max-error-pairs", type=int, default=20000)
    parser.add_argument("--skip-replacement", action="store_true")
    parser.add_argument("--save-cache", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Resolve and validate inputs without encoding data.")
    return parser.parse_args()


def existing_or_first(candidates: Sequence[Path]) -> Path:
    if not candidates:
        raise ValueError("No candidate paths provided")
    for path in candidates:
        expanded = path.expanduser()
        if expanded.is_file():
            return expanded.resolve()
    return candidates[0].expanduser().resolve()


def resolve_vq_checkpoint(kv_root: Path, requested: Optional[Path]) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    return existing_or_first(
        [
            REPO_ROOT / "checkpoints/codeflow_hml3d_release/rvq/part_vq_hml3d_overlap_best_top3.pth",
            kv_root / "checkpoints/vqvae_overlap_top3_20260529_hf/new_vq_overlap_top3_20260529_best_top3.pth",
            kv_root / "checkpoints/vqvae/net_best_fid.pth",
        ]
    )


def resolve_partition(kv_root: Path, requested: Optional[Path]) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    return existing_or_first(
        [
            REPO_ROOT / "checkpoints/codeflow_hml3d_release/rvq/skeleton_partition.json",
            kv_root / "checkpoints/vqvae_overlap_top3_20260529_hf/config/skeleton_partition.json",
            kv_root / "checkpoints/vqvae/skeleton_partition.json",
        ]
    )


def resolve_stats(kv_root: Path, data_root: Path, mean_path: Optional[Path], std_path: Optional[Path]) -> Tuple[Path, Path]:
    if mean_path is not None:
        mean = mean_path.expanduser().resolve()
    else:
        mean = existing_or_first(
            [
                REPO_ROOT / "checkpoints/codeflow_hml3d_release/stats/mean.npy",
                kv_root / "checkpoints/stats/mean.npy",
                data_root / "Mean.npy",
            ]
        )
    if std_path is not None:
        std = std_path.expanduser().resolve()
    else:
        std = existing_or_first(
            [
                REPO_ROOT / "checkpoints/codeflow_hml3d_release/stats/std.npy",
                kv_root / "checkpoints/stats/std.npy",
                data_root / "Std.npy",
            ]
        )
    return mean, std


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_formal_vq_model(
    kv_root: Path,
    checkpoint_path: Path,
    partition_path: Path,
    device: torch.device,
):
    diag.add_kv_control_to_path(kv_root)
    from kvctrl.models import vqvae

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VQ checkpoint not found: {checkpoint_path}")
    if not partition_path.is_file():
        raise FileNotFoundError(f"VQ partition not found: {partition_path}")

    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    partition_dataset = str(partition.get("dataset", "")).lower()
    feature_dim = int(partition.get("feature_dim", 251 if partition_dataset == "kit" else 263))
    dataname = "kit" if feature_dim == 251 or partition_dataset == "kit" else "t2m"

    cfg = dict(diag.VQ_CFG)
    cfg["dataname"] = dataname
    cfg["input_dim"] = feature_dim
    cfg["load_dir_vqvae"] = str(checkpoint_path)
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
    ckpt = load_torch_checkpoint(checkpoint_path)
    if isinstance(ckpt, dict) and "net" in ckpt:
        state_dict = ckpt["net"]
    elif isinstance(ckpt, dict) and "vq_model" in ckpt:
        state_dict = ckpt["vq_model"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"VQ load mismatch. missing={missing[:8]} unexpected={unexpected[:8]}")
    model.eval().to(device)

    if not hasattr(model, "vqvae") or not hasattr(model.vqvae, "quantizers"):
        raise RuntimeError("Loaded VQ model does not expose part quantizers")
    num_codes = int(model.vqvae.quantizers[0].codebook.shape[0])
    diag.VQ_CFG["nb_code"] = num_codes
    diag.VQ_CFG["dataname"] = dataname
    return model, partition, dataname, feature_dim


def formal_part_names(partition: Dict[str, object]) -> List[str]:
    n_parts = int(partition.get("n_parts", len(partition.get("partSeg", []))))
    names = partition.get("part_names")
    if isinstance(names, list) and len(names) == n_parts:
        return [str(name) for name in names]
    if n_parts == len(FORMAL_PART_NAMES):
        return list(FORMAL_PART_NAMES)
    return [f"part_{idx}" for idx in range(n_parts)]


def make_recover_joints_np(joints_num: int):
    def recover_joints_np(kv_root: Path, denorm_motion: np.ndarray, device: torch.device) -> np.ndarray:
        diag.add_kv_control_to_path(kv_root)
        from kvctrl.utils.motion_process import recover_from_ric

        with torch.no_grad():
            x = torch.from_numpy(denorm_motion).float().to(device)
            joints = recover_from_ric(x, joints_num).detach().cpu().numpy()
        return joints.astype(np.float32)

    return recover_joints_np


def write_partition_geometry(out_dir: Path, partition: Dict[str, object], part_names: Sequence[str]) -> None:
    payload = {
        "part_names": list(part_names),
        "joint_names": partition.get("joint_names"),
        "joint_labels": partition.get("joint_labels"),
        "part_joints_base": partition.get("part_joints_base"),
        "part_joints_overlap_sources": partition.get("part_joints_overlap_sources"),
        "part_joints_effective": partition.get("part_joints_effective"),
        "overlap_events": partition.get("overlap_events"),
    }
    (out_dir / "partition_geometry.json").write_text(
        json.dumps(payload, indent=2, default=diag.json_default) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rng = diag.seed_everything(args.seed)
    args.kv_root = args.kv_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_vq_checkpoint(args.kv_root, args.vq_checkpoint)
    partition_path = resolve_partition(args.kv_root, args.partition_file)
    mean_path, std_path = resolve_stats(args.kv_root, args.data_root, args.mean_path, args.std_path)

    device = torch.device(args.device)
    model, partition, dataname, feature_dim = load_formal_vq_model(
        args.kv_root,
        checkpoint_path,
        partition_path,
        device,
    )
    part_names = formal_part_names(partition)
    diag.PART_NAMES = list(part_names)
    diag.recover_joints_np = make_recover_joints_np(int(partition.get("joints_num", 22 if feature_dim == 263 else 21)))

    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    std = np.where(std == 0, 1.0, std)
    if mean.shape[-1] != feature_dim or std.shape[-1] != feature_dim:
        raise ValueError(
            f"Stats shape does not match partition feature_dim={feature_dim}: "
            f"mean={mean.shape} std={std.shape}"
        )

    part_seg = partition["partSeg"]
    if args.check_only:
        print(f"[check] checkpoint={checkpoint_path}")
        print(f"[check] partition={partition_path}")
        print(f"[check] mean={mean_path}")
        print(f"[check] std={std_path}")
        print(f"[check] dataname={dataname} feature_dim={feature_dim} parts={part_names}")
        return

    split_ids = diag.load_split_ids(args.data_root, args.split)
    encoded, proto_sums_by_space, counts, meta = diag.encode_dataset(
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
    codebooks = diag.codebooks_numpy(model)

    usage_rows = diag.compute_usage_rows(counts)
    corr_rows, nn_rows = diag.compute_distance_and_nn(
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
        stratified_rows = diag.compute_stratified_correlations(
            codebooks,
            proto_sums_by_space,
            counts,
            encoded.position_counts,
            args.min_code_count,
            args.max_pairs,
            rng,
        )
    bootstrap_rows = diag.compute_bootstrap_rows(
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
        error_cost_rows = diag.compute_error_cost_rows(
            codebooks,
            proto_sums_by_space,
            counts,
            args.min_code_count,
            args.max_error_pairs,
            rng,
        )
        error_cost_summary_rows = diag.summarize_error_cost(error_cost_rows)
    trajectory_rows = diag.compute_trajectory_rows(encoded, codebooks, rng)
    replacement_rows: List[Dict[str, object]] = []
    if not args.skip_replacement:
        replacement_rows = diag.compute_replacement_rows(
            model,
            encoded,
            codebooks,
            part_seg,
            args.replacement_samples,
            device,
            rng,
        )
    replacement_summary_rows = diag.summarize_replacement_rows(replacement_rows)

    summary = {
        **meta,
        "diagnostics_entry": str(Path(__file__).resolve()),
        "formal_part_names": list(part_names),
        "kv_root": str(args.kv_root),
        "data_root": str(args.data_root),
        "vq_checkpoint": str(checkpoint_path),
        "partition_file": str(partition_path),
        "mean_path": str(mean_path),
        "std_path": str(std_path),
        "dataname": dataname,
        "feature_dim": feature_dim,
        "split": args.split,
        "max_motions": args.max_motions,
        "min_code_count": args.min_code_count,
        "max_pairs": args.max_pairs,
        "bootstrap": args.bootstrap,
        "confound_controls": bool(args.confound_controls),
        "error_cost": bool(args.error_cost),
        "replacement_samples": args.replacement_samples,
        "code_usage": usage_rows,
        "replacement_summary": diag.summarize_replacement(replacement_rows),
    }

    diag.write_csv(args.out_dir / "code_usage.csv", usage_rows)
    diag.write_csv(args.out_dir / "distance_correlation.csv", corr_rows)
    diag.write_csv(args.out_dir / "nn_retrieval.csv", nn_rows)
    diag.write_csv(args.out_dir / "distance_correlation_stratified.csv", stratified_rows)
    diag.write_csv(args.out_dir / "bootstrap_ci.csv", bootstrap_rows)
    diag.write_csv(args.out_dir / "geometry_error_cost.csv", error_cost_rows)
    diag.write_csv(args.out_dir / "geometry_error_cost_summary.csv", error_cost_summary_rows)
    diag.write_csv(args.out_dir / "trajectory_smoothness.csv", trajectory_rows)
    diag.write_csv(args.out_dir / "replacement_near_mid_far.csv", replacement_rows)
    diag.write_csv(args.out_dir / "replacement_error_cost.csv", replacement_summary_rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=diag.json_default) + "\n",
        encoding="utf-8",
    )
    write_partition_geometry(args.out_dir, partition, part_names)

    if args.save_cache:
        np.savez_compressed(
            args.out_dir / "encoded_cache.npz",
            names=np.asarray(encoded.names),
            motions=encoded.motions,
            lengths=encoded.lengths,
            latent_valid=encoded.latent_valid,
            ids=encoded.ids,
            counts=counts,
            position_counts=encoded.position_counts,
        )

    diag.write_report(
        args.out_dir,
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
    print(f"[done] wrote formal diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
