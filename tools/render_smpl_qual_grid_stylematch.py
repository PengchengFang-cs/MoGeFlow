#!/usr/bin/env python3
"""Re-render the selected-three qualitative figure with SMPL body meshes.

Layout, camera, ground plane, trajectory arrows and temporal ordering follow
tools/render_motion_qualitative_grid.py (the stick version used in the paper);
only the human representation changes from a skeleton to the fitted SMPL mesh.

Stage 1 renders one PNG per grid cell, stage 2 crops them and composes the
labelled 3x3 panel that replaces 20260610-003631.png in the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.render_motion_qualitative_grid import keyframe_indices

SMPL_DIR = REPO_ROOT / "figures/qual_selected_three_20260609_smpl"
GEN_DIR = REPO_ROOT / "generation/qual_selected_three_20260609"

PROMPTS = [
    "A person crawls forward on hands and knees, then gets up and walks forward.",
    "A person crouches down, picks something up from the floor, then stands up and turns left.",
    "A person kneels on one knee, poses, then stands up and raises both arms.",
]
COLUMNS = [
    ("MoGeFlow", "ours_ps_cf", "pscf"),
    ("M-Transformer", "m_transformer", "mtransformer"),
    ("MoMask", "momask", "momask"),
]
ROW_TAGS = ["p0_crawl", "p1_pickup", "p2_kneel"]
SOURCE_JOINTS = {
    "pscf": [GEN_DIR / "ours_pscf_w200_bestfid_ema/joints" / f"sample{i:03d}_repeat00_len196_joints.npy" for i in range(3)],
    "mtransformer": [GEN_DIR / "mtransformer/joints" / f"sample{i}_repeat0_len196.npy" for i in range(3)],
    "momask": [GEN_DIR / "momask/joints" / f"sample{i:03d}_repeat00_len196.npy" for i in range(3)],
}

# plain SMPL grey, fully opaque
BODY_COLOR = "#b4b4b4"

# same trajectory colours as the stick figure
METHOD_COLORS = {
    "mogeflow": "#2ca25f",
    "ours": "#2ca25f",
    "ps-cf": "#2ca25f",
    "momask": "#c2410c",
}

ELEV, AZIM = 18.0, -64.0


def method_color(label: str) -> str:
    lowered = label.lower()
    for key, color in METHOD_COLORS.items():
        if key in lowered:
            return color
    return "#525252"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output_dir", type=Path, default=REPO_ROOT / "figures/qual_selected_three_20260820_smpl_stylematch")
    p.add_argument("--output_name", default="fig_qual_selected_three_smpl_labeled_compact")
    p.add_argument("--num_keyframes", type=int, default=7)
    p.add_argument("--cell_dpi", type=int, default=320)
    p.add_argument("--cell_size", type=float, default=5.0)
    p.add_argument("--fig_dpi", type=int, default=300)
    p.add_argument("--fig_width", type=float, default=11.4)
    p.add_argument("--font", default="STIXGeneral")
    p.add_argument("--only_cell", default=None, help="e.g. r0c0, for quick iteration")
    p.add_argument("--stage", choices=["cells", "compose", "all"], default="all")
    p.add_argument("--body_color", default=BODY_COLOR)
    p.add_argument("--floor_mode", choices=["median", "min"], default="median")
    return p.parse_args()


# ---------------------------------------------------------------- geometry


def prepare_layout(joints: np.ndarray, num_keyframes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Same placement rule as the stick figure: keyframes spread along the root path."""
    frames = keyframe_indices(len(joints), num_keyframes)
    root_xy = joints[:, 0, [0, 2]]
    path = root_xy[frames] - root_xy[frames[0]]
    path_extent = np.ptp(path, axis=0)
    path_scale = max(float(np.linalg.norm(path_extent)), 1e-6)
    if path_scale < 0.45:
        positions = np.stack(
            [np.linspace(-1.8, 1.8, len(frames)), np.zeros(len(frames), dtype=np.float32)], axis=1
        )
    else:
        positions = path / path_scale * 3.6
    return frames, positions.astype(np.float64)


def load_cell_meshes(stem: str, joints: np.ndarray, frames: np.ndarray, positions: np.ndarray, floor_mode: str):
    """Return (list of vertex arrays in axes coords [x, z, height], faces)."""
    mesh_dir = SMPL_DIR / "smpl_fit_iter08" / stem
    faces = None
    verts_list: List[np.ndarray] = []
    for k, (frame, pos) in enumerate(zip(frames, positions)):
        mesh = trimesh.load(mesh_dir / f"{k:04d}.ply", process=False)
        v = np.asarray(mesh.vertices, dtype=np.float64).copy()
        if faces is None:
            faces = np.asarray(mesh.faces, dtype=np.int64)
        root = joints[frame, 0, [0, 2]]
        v[:, 0] += pos[0] - root[0]
        v[:, 2] += pos[1] - root[1]
        verts_list.append(v)
    per_frame_min = np.array([float(v[:, 1].min()) for v in verts_list])
    y_floor = float(np.median(per_frame_min)) if floor_mode == "median" else float(per_frame_min.min())
    # axes convention of the stick figure: X = x, Y = z (depth), Z = height
    return [np.stack([v[:, 0], v[:, 2], v[:, 1] - y_floor], axis=1) for v in verts_list], faces


def cell_limits(verts_list: Sequence[np.ndarray]):
    pts = np.concatenate(verts_list, axis=0)
    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)
    pad_x = max(0.35, 0.08 * (x_max - x_min + 1e-6))
    pad_y = max(0.35, 0.08 * (y_max - y_min + 1e-6))
    return (
        (float(x_min - pad_x), float(x_max + pad_x)),
        (float(y_min - pad_y), float(y_max + pad_y)),
        (0.0, float(max(1.2, z_max * 1.08))),
    )


# ---------------------------------------------------------------- shading


def camera_dir(elev: float, azim: float) -> np.ndarray:
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def shaded_faces(verts: np.ndarray, faces: np.ndarray, base_rgb: np.ndarray):
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)

    cam = camera_dir(ELEV, AZIM)
    keep = (n @ cam) > -0.02  # back-face culling (outward normals face the camera)
    tri, n = tri[keep], n[keep]

    key = cam + np.array([0.15, 0.0, 0.95])
    key /= np.linalg.norm(key)
    fill = np.array([-cam[0], -cam[1], 0.35])
    fill /= np.linalg.norm(fill)

    lam = 0.42 + 0.58 * np.clip(n @ key, 0.0, 1.0) + 0.13 * np.clip(n @ fill, 0.0, 1.0)
    lam = np.clip(lam, 0.0, 1.25)[:, None]
    return tri, np.clip(base_rgb[None, :] * lam, 0.0, 1.0)


def draw_floor(ax, xlim, ylim) -> None:
    x0, x1 = xlim
    y0, y1 = ylim
    verts = [[(x0, y0, 0.0), (x0, y1, 0.0), (x1, y1, 0.0), (x1, y0, 0.0)]]
    ax.add_collection3d(
        Poly3DCollection(verts, facecolors=(0.78, 0.78, 0.78, 0.18), edgecolors=(0.55, 0.55, 0.55, 0.25), zorder=0)
    )


def draw_trajectory(ax, positions: np.ndarray, label: str) -> None:
    if len(positions) < 2:
        return
    color = method_color(label)
    ax.plot3D(positions[:, 0], positions[:, 1], np.zeros(len(positions)), color=color, linewidth=2.2, alpha=0.85, zorder=90)
    dx = float(positions[-1, 0] - positions[-2, 0])
    dy = float(positions[-1, 1] - positions[-2, 1])
    if abs(dx) + abs(dy) > 1e-4:
        ax.quiver(
            positions[-2, 0], positions[-2, 1], 0.02, dx, dy, 0.0,
            color=color, arrow_length_ratio=0.35, linewidth=2.2, normalize=False, zorder=91,
        )


def render_cell(ax, verts_list, faces, positions, label, body_color) -> None:
    xlim, ylim, zlim = cell_limits(verts_list)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    # explicit draw order: matplotlib's own collection sorting would mix the bodies
    try:
        ax.computed_zorder = False
    except Exception:
        pass

    draw_floor(ax, xlim, ylim)

    # one opaque collection per body, painted far -> near so occlusion is correct
    cam = camera_dir(ELEV, AZIM)
    depth = np.array([float(v.mean(axis=0) @ cam) for v in verts_list])
    base = np.array(to_rgb(body_color))
    for rank, idx in enumerate(np.argsort(depth)):
        tri, rgb = shaded_faces(verts_list[idx], faces, base)
        rgba = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
        coll = Poly3DCollection(
            tri, facecolors=rgba, edgecolors=rgba, linewidths=0.28, shade=False, zorder=10 + rank
        )
        ax.add_collection3d(coll)

    draw_trajectory(ax, positions, label)

    ax.view_init(elev=ELEV, azim=AZIM)
    try:
        ax.set_box_aspect((1.7, 1.0, 0.85))
    except Exception:
        pass
    ax.set_axis_off()


# ---------------------------------------------------------------- stages


def stage_cells(args) -> None:
    cells_dir = args.output_dir / "cells_panel"
    cells_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for row, tag in enumerate(ROW_TAGS):
        for col, (label, slug, prefix) in enumerate(COLUMNS):
            if args.only_cell and args.only_cell != f"r{row}c{col}":
                continue
            name = f"row{row:02d}_col{col:02d}_{slug}"
            stem = f"{prefix}_{tag}"
            joints = np.load(SOURCE_JOINTS[prefix][row]).astype(np.float64)
            frames, positions = prepare_layout(joints, args.num_keyframes)
            verts_list, faces = load_cell_meshes(stem, joints, frames, positions, args.floor_mode)

            fig = plt.figure(figsize=(args.cell_size, args.cell_size))
            ax = fig.add_subplot(111, projection="3d")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            render_cell(ax, verts_list, faces, positions, label, args.body_color)
            out = cells_dir / f"{name}.png"
            fig.savefig(out, dpi=args.cell_dpi, transparent=True, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig)
            meta.append({"row": row, "col": col, "stem": stem, "label": label, "cell": str(out)})
            print(f"cell {name} <- {stem}", flush=True)
    (args.output_dir / "cells_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def flatten_and_crop(path: Path) -> np.ndarray:
    """RGBA cell -> tightly cropped RGB over white (no alpha left to resample)."""
    img = mpimg.imread(path)
    if img.shape[-1] == 4:
        alpha = img[..., 3]
        rgb = img[..., :3] * alpha[..., None] + (1.0 - alpha[..., None])
    else:
        alpha = np.ones(img.shape[:2])
        rgb = img[..., :3]
    mask = alpha > 0.02
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) and len(cols):
        rgb = rgb[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return np.clip(rgb, 0.0, 1.0)


def stage_compose(args) -> None:
    cells_dir = args.output_dir / "cells_panel"
    cropped_dir = args.output_dir / "cells_panel_cropped"
    cropped_dir.mkdir(parents=True, exist_ok=True)

    images = {}
    for row in range(3):
        for col, (_, slug, _) in enumerate(COLUMNS):
            src = cells_dir / f"row{row:02d}_col{col:02d}_{slug}.png"
            img = flatten_and_crop(src)
            plt.imsave(cropped_dir / src.name, img)
            images[(row, col)] = img

    plt.rcParams.update(
        {"font.family": "serif", "font.serif": [args.font, "DejaVu Serif"], "mathtext.fontset": "stix"}
    )

    fig_w = args.fig_width
    cell_w = fig_w / 3.0
    row_h = [max(images[(r, c)].shape[0] / images[(r, c)].shape[1] for c in range(3)) * cell_w for r in range(3)]
    header_h = 0.36
    caption_h = 0.42
    fig_h = header_h + sum(row_h) + 3 * caption_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    y = fig_h - header_h
    for col, (label, _, _) in enumerate(COLUMNS):
        fig.text((col + 0.5) / 3.0, (y + 0.12) / fig_h, label, ha="center", va="bottom", fontsize=15.0, fontweight="bold")

    for row in range(3):
        y -= caption_h
        fig.text(0.5, (y + caption_h * 0.32) / fig_h, PROMPTS[row], ha="center", va="bottom", fontsize=11.5)
        y -= row_h[row]
        for col in range(3):
            img = images[(row, col)]
            h = img.shape[0] / img.shape[1] * cell_w
            ax = fig.add_axes([col / 3.0, y / fig_h, cell_w / fig_w, h / fig_h])
            ax.imshow(img, interpolation="antialiased", resample=True)
            ax.set_axis_off()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = args.output_dir / f"{args.output_name}.{ext}"
        fig.savefig(out, dpi=args.fig_dpi, facecolor="white")
        print(f"saved {out}", flush=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.stage in ("cells", "all"):
        stage_cells(args)
    if args.stage in ("compose", "all"):
        stage_compose(args)


if __name__ == "__main__":
    main()
