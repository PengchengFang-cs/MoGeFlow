"""Render static key-frame grids for text-to-motion qualitative figures.

The script consumes generated joint arrays and produces a paper-friendly
multi-row grid similar to qualitative motion figures in recent T2M papers.
It does not run inference; use gen_codeflow_t2m.py first.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paramUtil import t2m_kinematic_chain


CHAIN_COLORS = ["#2563eb", "#dc2626", "#333333", "#0891b2", "#d97706"]
METHOD_COLORS = {
    "ours": "#2ca25f",
    "ps-cf": "#2ca25f",
    "momask": "#c2410c",
}


@dataclass(frozen=True)
class Sample:
    caption: str
    joints_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--codeflow_results", type=Path, required=True)
    parser.add_argument("--codeflow_label", type=str, default="PS-CF (ours)")
    parser.add_argument("--momask_root", type=Path, default=None)
    parser.add_argument("--momask_label", type=str, default="MoMask")
    parser.add_argument("--output_dir", type=Path, default=Path("figures/qualitative_20260607"))
    parser.add_argument("--output_name", type=str, default="fig_qualitative_motion_grid")
    parser.add_argument("--num_keyframes", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--caption_width", type=int, default=88)
    parser.add_argument("--style", type=str, default="stick", choices=["stick", "mannequin"])
    return parser.parse_args()


def load_codeflow_samples(results_path: Path) -> List[Sample]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    root = results_path.parent
    samples = []
    for record in payload["records"]:
        if int(record.get("repeat", 0)) != 0:
            continue
        joints_path = Path(record["joints"])
        if not joints_path.is_absolute():
            joints_path = root / joints_path.relative_to(root) if str(joints_path).startswith(str(root)) else joints_path
        samples.append(Sample(caption=record["caption"], joints_path=joints_path))
    return samples


def load_momask_samples(root: Path, captions: Sequence[str]) -> List[Sample]:
    samples = []
    for sample_id, caption in enumerate(captions):
        joint_dir = root / "joints" / str(sample_id)
        candidates = sorted(joint_dir.glob(f"sample{sample_id}_repeat0_len*.npy"))
        candidates = [p for p in candidates if not p.name.endswith("_ik.npy")]
        if not candidates:
            raise FileNotFoundError(f"No MoMask joint file for sample {sample_id}: {joint_dir}")
        samples.append(Sample(caption=caption, joints_path=candidates[0]))
    return samples


def load_joints(path: Path) -> np.ndarray:
    joints = np.load(path)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints [T,J,3], got {joints.shape} from {path}")
    return joints.astype(np.float32)


def keyframe_indices(length: int, count: int) -> np.ndarray:
    count = max(2, min(count, length))
    return np.linspace(0, length - 1, count).round().astype(int)


def prepare_display(joints: np.ndarray, num_keyframes: int) -> Tuple[List[np.ndarray], np.ndarray]:
    data = joints.copy()
    data[:, :, 1] -= data[:, :, 1].min()

    frames = keyframe_indices(len(data), num_keyframes)
    root_xy = data[:, 0, [0, 2]]
    path = root_xy[frames] - root_xy[frames[0]]
    path_extent = np.ptp(path, axis=0)
    path_scale = max(float(np.linalg.norm(path_extent)), 1e-6)

    if path_scale < 0.45:
        positions = np.stack(
            [np.linspace(-1.8, 1.8, len(frames)), np.zeros(len(frames), dtype=np.float32)],
            axis=1,
        )
    else:
        positions = path / path_scale * 3.6

    poses = []
    for frame, pos in zip(frames, positions):
        pose = data[frame].copy()
        root = pose[0, [0, 2]].copy()
        pose[:, 0] -= root[0]
        pose[:, 2] -= root[1]
        pose[:, 0] += pos[0]
        pose[:, 2] += pos[1]
        poses.append(pose)
    return poses, positions


def bounds_for_row(prepared: Sequence[Tuple[List[np.ndarray], np.ndarray]]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    all_points = np.concatenate([np.concatenate(poses, axis=0) for poses, _ in prepared], axis=0)
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)
    pad_x = max(0.35, 0.08 * (x_max - x_min + 1e-6))
    pad_y = max(0.35, 0.08 * (z_max - z_min + 1e-6))
    return (
        (float(x_min - pad_x), float(x_max + pad_x)),
        (float(z_min - pad_y), float(z_max + pad_y)),
        (0.0, float(max(1.2, y_max * 1.12))),
    )


def draw_floor(ax, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    x0, x1 = xlim
    y0, y1 = ylim
    verts = [[(x0, y0, 0.0), (x0, y1, 0.0), (x1, y1, 0.0), (x1, y0, 0.0)]]
    floor = Poly3DCollection(verts, facecolors=(0.78, 0.78, 0.78, 0.18), edgecolors=(0.55, 0.55, 0.55, 0.25))
    ax.add_collection3d(floor)


def draw_pose(ax, pose: np.ndarray, alpha: float, linewidth: float) -> None:
    for chain, color in zip(t2m_kinematic_chain, CHAIN_COLORS):
        ax.plot3D(
            pose[chain, 0],
            pose[chain, 2],
            pose[chain, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            solid_capstyle="round",
        )


def skeleton_edges() -> List[Tuple[int, int]]:
    edges = []
    seen = set()
    for chain in t2m_kinematic_chain:
        for start, end in zip(chain[:-1], chain[1:]):
            edge = tuple(sorted((start, end)))
            if edge in seen:
                continue
            seen.add(edge)
            edges.append((start, end))
    return edges


SKELETON_EDGES = skeleton_edges()


def draw_pose_mannequin(ax, pose: np.ndarray, alpha: float, is_last: bool) -> None:
    color = "#111827" if is_last else "#6b7280"
    linewidth = 4.2 if is_last else 3.1
    joint_size = 15 if is_last else 8

    for start, end in SKELETON_EDGES:
        pts = pose[[start, end]]
        ax.plot3D(
            pts[:, 0],
            pts[:, 2],
            pts[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            solid_capstyle="round",
        )

    ax.scatter(
        pose[:, 0],
        pose[:, 2],
        pose[:, 1],
        color=color,
        alpha=min(0.95, alpha + 0.1),
        s=joint_size,
        depthshade=False,
        linewidths=0,
    )

    head_id = 15 if pose.shape[0] > 15 else pose.shape[0] - 1
    ax.scatter(
        [pose[head_id, 0]],
        [pose[head_id, 2]],
        [pose[head_id, 1]],
        color=color,
        alpha=min(1.0, alpha + 0.15),
        s=42 if is_last else 22,
        depthshade=False,
        linewidths=0,
    )


def method_color(label: str) -> str:
    lowered = label.lower()
    for key, color in METHOD_COLORS.items():
        if key in lowered:
            return color
    return "#525252"


def draw_cell(
    ax,
    poses: Sequence[np.ndarray],
    positions: np.ndarray,
    limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    label: str,
    style: str,
) -> None:
    xlim, ylim, zlim = limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    draw_floor(ax, xlim, ylim)

    if len(positions) > 1:
        color = method_color(label)
        traj_width = 2.4 if style == "mannequin" else 1.8
        ax.plot3D(positions[:, 0], positions[:, 1], np.zeros(len(positions)), color=color, linewidth=traj_width, alpha=0.82)
        dx = float(positions[-1, 0] - positions[-2, 0])
        dy = float(positions[-1, 1] - positions[-2, 1])
        if abs(dx) + abs(dy) > 1e-4:
            ax.quiver(
                positions[-2, 0],
                positions[-2, 1],
                0.02,
                dx,
                dy,
                0.0,
                color=color,
                arrow_length_ratio=0.35,
                linewidth=traj_width,
                normalize=False,
            )

    for idx, pose in enumerate(poses):
        alpha = 0.22 + 0.73 * (idx + 1) / len(poses)
        if style == "mannequin":
            draw_pose_mannequin(ax, pose, alpha=alpha, is_last=idx == len(poses) - 1)
        else:
            draw_pose(ax, pose, alpha=alpha, linewidth=1.2 if idx < len(poses) - 1 else 1.8)

    ax.view_init(elev=18, azim=-64)
    try:
        ax.set_box_aspect((1.7, 1.0, 0.85))
    except Exception:
        pass
    ax.set_axis_off()


def wrapped_caption(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(f"\"{text}\"", width=width, break_long_words=False))


def render(
    methods: Dict[str, List[Sample]],
    output_dir: Path,
    output_name: str,
    num_keyframes: int,
    dpi: int,
    caption_width: int,
    style: str = "stick",
) -> None:
    labels = list(methods.keys())
    captions = [sample.caption for sample in next(iter(methods.values()))]
    num_rows = len(captions)
    num_cols = len(labels)

    fig = plt.figure(figsize=(4.35 * num_cols, 2.95 * num_rows), constrained_layout=False)
    height_ratios: List[float] = []
    for _ in range(num_rows):
        height_ratios.extend([0.18, 1.0])
    grid = fig.add_gridspec(
        num_rows * 2,
        num_cols,
        height_ratios=height_ratios,
        left=0.025,
        right=0.995,
        bottom=0.025,
        top=0.88,
        wspace=0.035,
        hspace=0.02,
    )

    for col, label in enumerate(labels):
        x = (col + 0.5) / num_cols
        fig.text(x, 0.945, label, ha="center", va="top", fontsize=12, fontweight="bold")

    for row, caption in enumerate(captions):
        caption_ax = fig.add_subplot(grid[row * 2, :])
        caption_ax.set_axis_off()
        caption_ax.text(
            0.5,
            0.45,
            wrapped_caption(caption, caption_width),
            ha="center",
            va="center",
            fontsize=8.8,
            fontstyle="italic",
        )

        row_prepared = []
        for label in labels:
            joints = load_joints(methods[label][row].joints_path)
            row_prepared.append(prepare_display(joints, num_keyframes))

        for col, label in enumerate(labels):
            ax = fig.add_subplot(grid[row * 2 + 1, col], projection="3d")
            poses, positions = row_prepared[col]
            cell_limits = bounds_for_row([row_prepared[col]])
            draw_cell(ax, poses, positions, cell_limits, label, style=style)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = output_dir / f"{output_name}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"saved {out}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    codeflow_samples = load_codeflow_samples(args.codeflow_results)
    methods: Dict[str, List[Sample]] = {args.codeflow_label: codeflow_samples}

    if args.momask_root is not None:
        captions = [sample.caption for sample in codeflow_samples]
        methods[args.momask_label] = load_momask_samples(args.momask_root, captions)

    lengths = {label: len(samples) for label, samples in methods.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Method sample counts differ: {lengths}")

    render(methods, args.output_dir, args.output_name, args.num_keyframes, args.dpi, args.caption_width, style=args.style)


if __name__ == "__main__":
    main()
