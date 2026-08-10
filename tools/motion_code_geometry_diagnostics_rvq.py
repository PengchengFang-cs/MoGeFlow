#!/usr/bin/env python3
"""Geometry diagnostics for the generic MoMask RVQ tokenizer.

This is a separate entry point from ``motion_code_geometry_diagnostics.py``,
following the precedent of ``motion_code_geometry_diagnostics_formal.py``: the
original script stays untouched (its kv_part path is frozen), and this file
adapts the frozen MoMask residual VQ-VAE -- the tokenizer the June
"Generic RVQ + flow" ablation trained against
(``checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2``) -- to the same
diagnostics protocol.

The "part" axis of the diagnostics becomes the residual-quantizer layer axis:
every layer sees the full-body feature patch, so partSeg is the full feature
range for each layer and prototypes measure what motion content each layer's
codes correspond to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(TOOLS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import motion_code_geometry_diagnostics as diag
from models.codeflow.momask_vq import MoMaskRVQTokenizer


DEFAULT_RVQ_ROOT = REPO_ROOT / "checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2"
DEFAULT_DATA_ROOT = REPO_ROOT / "dataset/HumanML3D"
DEFAULT_OUT_DIR = REPO_ROOT / "eval_results/diag_20260809/rvq_geometry"
# Only needed for joint_pos/joint_pos_vel prototype spaces (recover_from_ric).
DEFAULT_KV_ROOT = Path("/scratch/pf2m24/projects/Umdd/KV-Control")

HUMANML3D_FEATURE_DIM = 263
HUMANML3D_JOINTS = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run geometry diagnostics for the generic MoMask RVQ codebooks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vq-checkpoint", type=Path, default=DEFAULT_RVQ_ROOT / "model" / "net_best_fid.tar")
    parser.add_argument("--vq-opt-path", type=Path, default=None, help="Defaults to <ckpt>/../../opt.txt.")
    parser.add_argument("--mean-path", type=Path, default=DEFAULT_RVQ_ROOT / "meta" / "mean.npy")
    parser.add_argument("--std-path", type=Path, default=DEFAULT_RVQ_ROOT / "meta" / "std.npy")
    parser.add_argument("--kv-root", type=Path, default=DEFAULT_KV_ROOT, help="Only used for joint-space prototype recovery.")
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
    parser.add_argument("--check-only", action="store_true", help="Resolve and validate inputs without encoding data.")
    return parser.parse_args()


class _RVQQuantizerView:
    """Exposes ``.codebook`` the way the diagnostics read part quantizers."""

    def __init__(self, codebook: torch.Tensor) -> None:
        self.codebook = codebook


class _RVQVAEView:
    """Duck-typed ``model.vqvae`` view: ``.quantizers`` plus ``forward_decoder``."""

    def __init__(self, tokenizer: MoMaskRVQTokenizer) -> None:
        self._tokenizer = tokenizer
        self.quantizers = [
            _RVQQuantizerView(tokenizer.codebooks[idx]) for idx in range(tokenizer.num_quantizers)
        ]

    def forward_decoder(self, ids_grid: torch.Tensor) -> torch.Tensor:
        return self._tokenizer.decode_ids(ids_grid.to(self._tokenizer.device))


class MoMaskRVQDiagnosticsAdapter:
    """Adapts MoMaskRVQTokenizer to the encode/decode surface the diagnostics use.

    ``model(x, type="encode")`` must return part-major flattened ids [B, Q*T]
    (what ``ids_flat_to_grid`` expects); the residual-layer axis plays the part
    axis.
    """

    def __init__(self, tokenizer: MoMaskRVQTokenizer) -> None:
        self._tokenizer = tokenizer
        self.vqvae = _RVQVAEView(tokenizer)
        self.num_quantizers = tokenizer.num_quantizers
        self.num_codes = tokenizer.num_codes

    def __call__(self, x: torch.Tensor, type: str = "encode") -> torch.Tensor:
        if type != "encode":
            raise ValueError(f'MoMaskRVQDiagnosticsAdapter only supports type="encode", got {type!r}')
        ids_grid = self._tokenizer.encode_ids(x)  # [B, T, Q]
        bsz, latent_len, num_q = ids_grid.shape
        return ids_grid.permute(0, 2, 1).reshape(bsz, num_q * latent_len)


def synthetic_rvq_partition(num_quantizers: int) -> Dict[str, object]:
    """Every residual layer covers the full body: full feature range, all joints."""
    return {
        "dataset": "t2m",
        "feature_dim": HUMANML3D_FEATURE_DIM,
        "n_parts": num_quantizers,
        "part_names": [f"rvq_layer_{idx}" for idx in range(num_quantizers)],
        "partSeg": [list(range(HUMANML3D_FEATURE_DIM)) for _ in range(num_quantizers)],
        "part_joints_effective": [list(range(HUMANML3D_JOINTS)) for _ in range(num_quantizers)],
    }


def load_rvq_adapter(
    checkpoint_path: Path,
    opt_path: Optional[Path],
    device: torch.device,
) -> Tuple[MoMaskRVQDiagnosticsAdapter, MoMaskRVQTokenizer]:
    tokenizer = MoMaskRVQTokenizer(
        checkpoint_path=str(checkpoint_path),
        opt_path=str(opt_path) if opt_path else None,
        target_mode="stage",
        device=device,
    )
    return MoMaskRVQDiagnosticsAdapter(tokenizer), tokenizer


def main() -> None:
    args = parse_args()
    rng = diag.seed_everything(args.seed)
    args.data_root = args.data_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.kv_root = args.kv_root.expanduser()
    checkpoint_path = args.vq_checkpoint.expanduser().resolve()
    mean_path = args.mean_path.expanduser().resolve()
    std_path = args.std_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RVQ checkpoint not found: {checkpoint_path}")
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(f"RVQ stats not found: {mean_path} / {std_path}")

    device = torch.device(args.device)
    adapter, tokenizer = load_rvq_adapter(checkpoint_path, args.vq_opt_path, device)
    partition = synthetic_rvq_partition(tokenizer.num_quantizers)
    part_seg = partition["partSeg"]

    # Scope the module-level knobs of the shared diagnostics to this tokenizer,
    # exactly as the formal entry point does for the overlap partition.
    diag.VQ_CFG["nb_code"] = int(tokenizer.num_codes)
    diag.VQ_CFG["dataname"] = "t2m"
    diag.PART_NAMES = list(partition["part_names"])

    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    std = np.where(std == 0, 1.0, std)
    if mean.shape[-1] != HUMANML3D_FEATURE_DIM or std.shape[-1] != HUMANML3D_FEATURE_DIM:
        raise ValueError(f"Expected {HUMANML3D_FEATURE_DIM}-dim stats, got mean={mean.shape} std={std.shape}")

    if args.check_only:
        print(f"[check] checkpoint={checkpoint_path}")
        print(f"[check] opt={tokenizer.opt_path}")
        print(f"[check] mean={mean_path}")
        print(f"[check] std={std_path}")
        print(
            f"[check] num_quantizers={tokenizer.num_quantizers} nb_code={tokenizer.num_codes} "
            f"code_dim={tokenizer.code_dim} parts={diag.PART_NAMES}"
        )
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_ids = diag.load_split_ids(args.data_root, args.split)
    encoded, proto_sums_by_space, counts, meta = diag.encode_dataset(
        adapter,
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
    codebooks = diag.codebooks_numpy(adapter)

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
            adapter,
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
        "backend": "momask_rvq",
        "kv_root": str(args.kv_root),
        "data_root": str(args.data_root),
        "vq_checkpoint": str(checkpoint_path),
        "vq_opt_path": str(tokenizer.opt_path),
        "mean_path": str(mean_path),
        "std_path": str(std_path),
        "partition_file": "synthetic_full_body_per_rvq_layer",
        "num_quantizers": int(tokenizer.num_quantizers),
        "nb_code": int(tokenizer.num_codes),
        "code_dim": int(tokenizer.code_dim),
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
        json.dumps(summary, indent=2, default=diag.json_default) + "\n", encoding="utf-8"
    )

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
    print(f"[done] wrote RVQ geometry diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
