#!/usr/bin/env python3
"""Diagnose KIT-ML motion length handling.

This script is intentionally independent from the model/tokenizer diagnostics.
It checks the data-side contract used by the current training and evaluation
code:

- Text-to-motion filtering keeps KIT motions with 24 <= length < 200.
- Generated/evaluated motions are padded/cropped to max_motion_length=196.
- CodeFlow token lengths use unit_length=4, so the max valid token length is 49.
- VQ tokenizer training uses fixed 64-frame windows, a different length from
  the generation/evaluation length.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


DEFAULT_DATA_ROOT_CANDIDATES = [
    Path("/scratch/pf2m24/data/KIT-ML"),
    Path("/iridisfs/scratch/pf2m24/data/KIT-ML"),
    Path("dataset/KIT-ML"),
]
DEFAULT_OUT_DIR = Path("geometry_diagnostics/kit_motion_length")


@dataclass
class MotionRecord:
    split: str
    name: str
    status: str
    raw_length: Optional[int] = None
    feature_dim: Optional[int] = None
    reason: str = ""
    t2m_accepted: bool = False
    length_lt_min: bool = False
    length_ge_upper: bool = False
    aligned_single: Optional[int] = None
    aligned_double: Optional[int] = None
    token_single: Optional[int] = None
    token_double: Optional[int] = None
    pad_single: Optional[int] = None
    pad_double: Optional[int] = None
    vq_usable: bool = False
    vq_short_lt_window: bool = False
    vq_eq_window: bool = False
    vq_windows_current: int = 0
    vq_windows_inclusive: int = 0


@dataclass
class TextSegmentRecord:
    split: str
    name: str
    line_index: int
    status: str
    caption: str = ""
    f_tag: Optional[float] = None
    to_tag: Optional[float] = None
    current_code_frames: Optional[int] = None
    kit_fps_frames: Optional[int] = None
    current_code_accepted: Optional[bool] = None
    kit_fps_accepted: Optional[bool] = None
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write read-only diagnostics for KIT-ML motion length handling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=None, help="KIT-ML root containing split files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "train_val", "all"])
    parser.add_argument("--max-motion-length", type=int, default=196)
    parser.add_argument("--unit-length", type=int, default=4)
    parser.add_argument("--min-motion-len", type=int, default=24)
    parser.add_argument("--filter-upper", type=int, default=200, help="Exclusive upper length filter.")
    parser.add_argument("--vq-window-size", type=int, default=64)
    parser.add_argument(
        "--segment-code-fps",
        type=float,
        default=20.0,
        help="FPS multiplier used by the current text segment slicing code.",
    )
    parser.add_argument("--kit-fps", type=float, default=12.5, help="Physical KIT frame rate.")
    parser.add_argument(
        "--write-all-text-lines",
        action="store_true",
        help="Write every text line to text_segments.csv. By default only bad/nonzero-tag lines are written.",
    )
    return parser.parse_args()


def resolve_data_root(requested: Optional[Path]) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"KIT-ML data root not found: {path}")
        return path
    for candidate in DEFAULT_DATA_ROOT_CANDIDATES:
        path = candidate.expanduser().resolve()
        if path.is_dir():
            return path
    tried = ", ".join(str(p) for p in DEFAULT_DATA_ROOT_CANDIDATES)
    raise FileNotFoundError(f"KIT-ML data root not found. Tried: {tried}")


def selected_splits(selection: str) -> List[str]:
    if selection == "train_val":
        return ["train", "val"]
    if selection == "all":
        return ["train", "val", "test"]
    return [selection]


def load_split_ids(data_root: Path, split: str) -> List[str]:
    split_path = data_root / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_shape(path: Path) -> tuple[Optional[int], Optional[int], str]:
    try:
        array = np.load(path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 - diagnostics should keep going per-file.
        return None, None, f"load_error: {exc}"
    if array.ndim != 2:
        return None, None, f"invalid_ndim: {array.ndim}"
    return int(array.shape[0]), int(array.shape[1]), ""


def aligned_lengths(
    raw_length: int,
    max_motion_length: int,
    unit_length: int,
    min_motion_len: int,
) -> tuple[int, int]:
    min_aligned = int(math.ceil(min_motion_len / float(unit_length)) * unit_length)
    max_aligned = (max_motion_length // unit_length) * unit_length
    max_available = (raw_length // unit_length) * unit_length
    single = (raw_length // unit_length) * unit_length
    double = (raw_length // unit_length - 1) * unit_length
    single = min(single, max_aligned, max_available)
    double = min(double, max_aligned, max_available)
    single = max(single, min_aligned)
    double = max(double, min_aligned)
    return int(single), int(double)


def diagnose_motion(
    split: str,
    name: str,
    data_root: Path,
    args: argparse.Namespace,
) -> MotionRecord:
    motion_path = data_root / "new_joint_vecs" / f"{name}.npy"
    if not motion_path.is_file():
        return MotionRecord(split=split, name=name, status="missing", reason="motion file missing")

    raw_length, feature_dim, reason = safe_shape(motion_path)
    if raw_length is None:
        return MotionRecord(split=split, name=name, status="invalid", feature_dim=feature_dim, reason=reason)

    length_lt_min = raw_length < args.min_motion_len
    length_ge_upper = raw_length >= args.filter_upper
    accepted = not length_lt_min and not length_ge_upper

    record = MotionRecord(
        split=split,
        name=name,
        status="ok",
        raw_length=raw_length,
        feature_dim=feature_dim,
        t2m_accepted=accepted,
        length_lt_min=length_lt_min,
        length_ge_upper=length_ge_upper,
        vq_usable=raw_length >= args.vq_window_size,
        vq_short_lt_window=raw_length < args.vq_window_size,
        vq_eq_window=raw_length == args.vq_window_size,
        vq_windows_current=max(raw_length - args.vq_window_size, 0),
        vq_windows_inclusive=max(raw_length - args.vq_window_size + 1, 0),
    )

    if accepted:
        single, double = aligned_lengths(
            raw_length,
            args.max_motion_length,
            args.unit_length,
            args.min_motion_len,
        )
        record.aligned_single = single
        record.aligned_double = double
        record.token_single = single // args.unit_length
        record.token_double = double // args.unit_length
        record.pad_single = args.max_motion_length - single
        record.pad_double = args.max_motion_length - double
    return record


def parse_float_tag(value: str) -> float:
    tag = float(value)
    if np.isnan(tag):
        return 0.0
    return tag


def segment_frames(f_tag: float, to_tag: float, fps: float) -> int:
    return int(to_tag * fps) - int(f_tag * fps)


def diagnose_text_file(
    split: str,
    name: str,
    data_root: Path,
    args: argparse.Namespace,
) -> List[TextSegmentRecord]:
    text_path = data_root / "texts" / f"{name}.txt"
    if not text_path.is_file():
        return [
            TextSegmentRecord(
                split=split,
                name=name,
                line_index=-1,
                status="missing",
                reason="text file missing",
            )
        ]

    rows: List[TextSegmentRecord] = []
    for idx, line in enumerate(text_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.strip():
            continue
        parts = line.strip().split("#")
        if len(parts) < 4:
            rows.append(
                TextSegmentRecord(
                    split=split,
                    name=name,
                    line_index=idx,
                    status="bad",
                    caption=parts[0] if parts else "",
                    reason=f"expected at least 4 #-separated fields, got {len(parts)}",
                )
            )
            continue
        try:
            f_tag = parse_float_tag(parts[2])
            to_tag = parse_float_tag(parts[3])
        except ValueError as exc:
            rows.append(
                TextSegmentRecord(
                    split=split,
                    name=name,
                    line_index=idx,
                    status="bad",
                    caption=parts[0],
                    reason=f"bad tag: {exc}",
                )
            )
            continue

        current_frames = segment_frames(f_tag, to_tag, args.segment_code_fps)
        kit_frames = segment_frames(f_tag, to_tag, args.kit_fps)
        nonzero = f_tag != 0.0 or to_tag != 0.0
        status = "segment" if nonzero else "full_motion"
        row = TextSegmentRecord(
            split=split,
            name=name,
            line_index=idx,
            status=status,
            caption=parts[0],
            f_tag=f_tag,
            to_tag=to_tag,
            current_code_frames=current_frames,
            kit_fps_frames=kit_frames,
            current_code_accepted=args.min_motion_len <= current_frames < args.filter_upper,
            kit_fps_accepted=args.min_motion_len <= kit_frames < args.filter_upper,
        )
        if args.write_all_text_lines or nonzero:
            rows.append(row)
    return rows


def percentile(values: Sequence[int], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def int_or_none(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def summarize_records(split: str, ids: Sequence[str], records: Sequence[MotionRecord], text_rows: Sequence[TextSegmentRecord]) -> Dict[str, object]:
    loaded = [r for r in records if r.raw_length is not None]
    accepted = [r for r in loaded if r.t2m_accepted]
    lengths = [int(r.raw_length) for r in loaded if r.raw_length is not None]
    accepted_lengths = [int(r.raw_length) for r in accepted if r.raw_length is not None]
    token_single = [int(r.token_single) for r in accepted if r.token_single is not None]
    pad_single = [int(r.pad_single) for r in accepted if r.pad_single is not None]
    bad_text = [r for r in text_rows if r.status in {"bad", "missing"}]
    nonzero_text = [r for r in text_rows if r.status == "segment"]

    return {
        "split": split,
        "n_ids": len(ids),
        "n_loaded_files": len(loaded),
        "n_missing_motion_files": sum(r.status == "missing" for r in records),
        "n_invalid_motion_files": sum(r.status == "invalid" for r in records),
        "raw_min": int_or_none(percentile(lengths, 0)),
        "raw_p25": percentile(lengths, 25),
        "raw_median": percentile(lengths, 50),
        "raw_p75": percentile(lengths, 75),
        "raw_max": int_or_none(percentile(lengths, 100)),
        "raw_mean": float(np.mean(lengths)) if lengths else None,
        "n_lt_min_motion_len": sum(r.length_lt_min for r in loaded),
        "n_ge_filter_upper": sum(r.length_ge_upper for r in loaded),
        "n_t2m_accepted": len(accepted),
        "accepted_raw_min": int_or_none(percentile(accepted_lengths, 0)),
        "accepted_raw_median": percentile(accepted_lengths, 50),
        "accepted_raw_max": int_or_none(percentile(accepted_lengths, 100)),
        "token_single_min": int_or_none(percentile(token_single, 0)),
        "token_single_median": percentile(token_single, 50),
        "token_single_max": int_or_none(percentile(token_single, 100)),
        "pad_single_mean": float(np.mean(pad_single)) if pad_single else None,
        "n_vq_short_lt_window": sum(r.vq_short_lt_window for r in loaded),
        "n_vq_eq_window": sum(r.vq_eq_window for r in loaded),
        "n_vq_usable_ge_window": sum(r.vq_usable for r in loaded),
        "vq_windows_current_total": sum(r.vq_windows_current for r in loaded),
        "vq_windows_inclusive_total": sum(r.vq_windows_inclusive for r in loaded),
        "n_text_rows_written": len(text_rows),
        "n_text_nonzero_segments": len(nonzero_text),
        "n_text_bad_or_missing": len(bad_text),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(
    out_dir: Path,
    data_root: Path,
    args: argparse.Namespace,
    summaries: Sequence[Dict[str, object]],
    aggregate: Dict[str, object],
) -> None:
    max_tokens = args.max_motion_length // args.unit_length
    lines = [
        "# KIT Motion Length Diagnostics",
        "",
        f"- Data root: `{data_root}`",
        f"- Selected split: `{args.split}`",
        f"- Generation/eval max motion length: `{args.max_motion_length}` frames",
        f"- Unit length: `{args.unit_length}` frames, max token length `{max_tokens}`",
        f"- KIT T2M filter: `{args.min_motion_len} <= length < {args.filter_upper}`",
        f"- VQ training window: `{args.vq_window_size}` frames",
        "",
        "## Split Summary",
        "",
        "| split | ids | loaded | missing | raw median | raw max | accepted | <min | >=upper | <VQ window | VQ windows | token median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {split} | {n_ids} | {n_loaded_files} | {n_missing_motion_files} | {raw_median} | {raw_max} | "
            "{n_t2m_accepted} | {n_lt_min_motion_len} | {n_ge_filter_upper} | {n_vq_short_lt_window} | "
            "{vq_windows_current_total} | {token_single_median} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Selected Aggregate",
            "",
            f"- Loaded motion files: `{aggregate['n_loaded_files']}` / `{aggregate['n_ids']}` ids",
            f"- T2M accepted motions: `{aggregate['n_t2m_accepted']}`",
            f"- Filtered by short length: `{aggregate['n_lt_min_motion_len']}`",
            f"- Filtered by upper bound: `{aggregate['n_ge_filter_upper']}`",
            f"- Shorter than VQ window: `{aggregate['n_vq_short_lt_window']}`",
            f"- Current VQ-window sample count: `{aggregate['vq_windows_current_total']}`",
            f"- Inclusive VQ-window sample count for comparison: `{aggregate['vq_windows_inclusive_total']}`",
            f"- Nonzero text segments written: `{aggregate['n_text_nonzero_segments']}`",
            f"- Bad/missing text rows written: `{aggregate['n_text_bad_or_missing']}`",
            "",
            "## Interpretation",
            "",
            "- The standard KIT generation/evaluation length is 196 frames.",
            "- The VQ training length is the separate 64-frame fixed window used for tokenizer reconstruction training.",
            "- With unit_length=4, 196 frames correspond to 49 valid CodeFlow tokens.",
            "- The current fixed-window VQ dataset contributes `length - window_size` samples per loaded motion, so motions of exactly 64 frames contribute zero indexed samples.",
            "",
            "## Outputs",
            "",
            "- `summary.json`: options and split summaries.",
            "- `split_summary.csv`: one row per split plus the selected aggregate.",
            "- `motion_lengths.csv`: one row per split id with filtering, padding, token, and VQ-window diagnostics.",
            "- `text_segments.csv`: nonzero-tag/bad text rows, or all text rows when `--write-all-text-lines` is set.",
        ]
    )
    (out_dir / "KIT_MOTION_LENGTH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_to_ids: Dict[str, List[str]] = {}
    all_records: List[MotionRecord] = []
    all_text_rows: List[TextSegmentRecord] = []
    summaries: List[Dict[str, object]] = []

    for split in selected_splits(args.split):
        ids = load_split_ids(data_root, split)
        split_to_ids[split] = ids
        records = [diagnose_motion(split, name, data_root, args) for name in ids]
        text_rows: List[TextSegmentRecord] = []
        for name in ids:
            text_rows.extend(diagnose_text_file(split, name, data_root, args))
        summaries.append(summarize_records(split, ids, records, text_rows))
        all_records.extend(records)
        all_text_rows.extend(text_rows)

    aggregate_ids = [name for ids in split_to_ids.values() for name in ids]
    aggregate = summarize_records("selected", aggregate_ids, all_records, all_text_rows)
    summary_payload = {
        "data_root": str(data_root),
        "options": {
            "split": args.split,
            "max_motion_length": args.max_motion_length,
            "unit_length": args.unit_length,
            "min_motion_len": args.min_motion_len,
            "filter_upper": args.filter_upper,
            "vq_window_size": args.vq_window_size,
            "segment_code_fps": args.segment_code_fps,
            "kit_fps": args.kit_fps,
            "write_all_text_lines": args.write_all_text_lines,
        },
        "split_summaries": summaries,
        "selected_aggregate": aggregate,
    }

    motion_rows = [asdict(record) for record in all_records]
    text_rows = [asdict(row) for row in all_text_rows]
    split_rows = list(summaries) + [aggregate]

    write_csv(out_dir / "motion_lengths.csv", motion_rows, list(MotionRecord.__dataclass_fields__.keys()))
    write_csv(out_dir / "text_segments.csv", text_rows, list(TextSegmentRecord.__dataclass_fields__.keys()))
    write_csv(out_dir / "split_summary.csv", split_rows, list(aggregate.keys()))
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, data_root, args, summaries, aggregate)

    print(f"Wrote KIT motion length diagnostics to {out_dir}")
    print(
        "Selected aggregate: "
        f"{aggregate['n_t2m_accepted']} accepted / {aggregate['n_loaded_files']} loaded, "
        f"{aggregate['n_vq_short_lt_window']} shorter than VQ window, "
        f"{aggregate['vq_windows_current_total']} current VQ windows."
    )


if __name__ == "__main__":
    main()
