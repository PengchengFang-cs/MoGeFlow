#!/usr/bin/env python3
"""Render paged qualitative candidate grids from several results.json files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.render_motion_qualitative_grid import Sample, render


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
        joints_path = Path(record["joints"])
        samples.append(Sample(caption=record["caption"], joints_path=joints_path))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_prefix", default="fig_candidates")
    parser.add_argument("--rows_per_page", type=int, default=4)
    parser.add_argument("--num_keyframes", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--caption_width", type=int, default=95)
    parser.add_argument("--style", choices=["stick", "mannequin"], default="mannequin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods: Dict[str, List[Sample]] = {label: load_results(path) for label, path in args.method}
    counts = {label: len(samples) for label, samples in methods.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Method sample counts differ: {counts}")

    total = next(iter(counts.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    page = 0
    for start in range(0, total, int(args.rows_per_page)):
        page += 1
        stop = min(total, start + int(args.rows_per_page))
        chunk = {label: samples[start:stop] for label, samples in methods.items()}
        render(
            chunk,
            output_dir=args.output_dir,
            output_name=f"{args.output_prefix}_page{page:02d}",
            num_keyframes=int(args.num_keyframes),
            dpi=int(args.dpi),
            caption_width=int(args.caption_width),
            style=str(args.style),
        )


if __name__ == "__main__":
    main()
