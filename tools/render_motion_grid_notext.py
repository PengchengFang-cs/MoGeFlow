#!/usr/bin/env python3
"""Render a text-free qualitative joint grid for paper composition."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.render_motion_qualitative_grid import Sample, bounds_for_row, draw_cell, load_joints, prepare_display


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Method must be LABEL=/path/to/results.json")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Method label cannot be empty")
    return label, Path(path)


def load_results(results_path: Path) -> List[Sample]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    samples: List[Sample] = []
    for record in payload["records"]:
        if int(record.get("repeat", 0)) != 0:
            continue
        samples.append(Sample(caption=record["caption"], joints_path=Path(record["joints"])))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", default="fig_motion_grid_notext")
    parser.add_argument("--num_keyframes", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--style", choices=["stick", "mannequin"], default="stick")
    parser.add_argument("--cell_width", type=float, default=3.85)
    parser.add_argument("--cell_height", type=float, default=2.45)
    parser.add_argument("--pad", type=float, default=0.018)
    parser.add_argument("--wspace", type=float, default=0.035)
    parser.add_argument("--hspace", type=float, default=0.025)
    parser.add_argument("--save_cells", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return value or "method"


def render_notext(
    methods: Dict[str, List[Sample]],
    output_dir: Path,
    output_name: str,
    num_keyframes: int,
    dpi: int,
    style: str,
    cell_width: float,
    cell_height: float,
    pad: float,
    wspace: float,
    hspace: float,
    save_cells: bool,
) -> None:
    labels = list(methods.keys())
    counts = {label: len(samples) for label, samples in methods.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Method sample counts differ: {counts}")

    num_rows = next(iter(counts.values()))
    num_cols = len(labels)
    fig = plt.figure(figsize=(cell_width * num_cols, cell_height * num_rows), constrained_layout=False)
    grid = fig.add_gridspec(
        num_rows,
        num_cols,
        left=pad,
        right=1.0 - pad,
        bottom=pad,
        top=1.0 - pad,
        wspace=wspace,
        hspace=hspace,
    )

    cell_dir = output_dir / "cells_panel"
    if save_cells:
        cell_dir.mkdir(parents=True, exist_ok=True)

    for row in range(num_rows):
        row_prepared = []
        for label in labels:
            joints = load_joints(methods[label][row].joints_path)
            row_prepared.append(prepare_display(joints, num_keyframes))
        row_limits = bounds_for_row(row_prepared)

        for col, label in enumerate(labels):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            poses, positions = row_prepared[col]
            draw_cell(ax, poses, positions, row_limits, label, style=style)
            if save_cells:
                cell_fig = plt.figure(figsize=(cell_width, cell_height), constrained_layout=False)
                cell_ax = cell_fig.add_subplot(111, projection="3d")
                draw_cell(cell_ax, poses, positions, row_limits, label, style=style)
                cell_out = cell_dir / f"row{row:02d}_col{col:02d}_{slugify(label)}.png"
                cell_fig.savefig(cell_out, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
                plt.close(cell_fig)
                print(f"saved {cell_out}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"{output_name}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
        print(f"saved {out}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    methods = {label: load_results(path) for label, path in args.method}
    render_notext(
        methods,
        args.output_dir,
        args.output_name,
        int(args.num_keyframes),
        int(args.dpi),
        str(args.style),
        float(args.cell_width),
        float(args.cell_height),
        float(args.pad),
        float(args.wspace),
        float(args.hspace),
        bool(args.save_cells),
    )


if __name__ == "__main__":
    main()
