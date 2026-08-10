#!/usr/bin/env python3
"""Build the formal six-part overlap skeleton partition JSON.

The HumanML3D specification in this file is constrained to reproduce the
released overlap-top3 tokenizer partition exactly for the core fields consumed
by the VQ and diagnostics. The KIT specification is the same part order and
boundary-overlap rule mapped onto the 21-joint KIT skeleton.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PART_NAMES = [
    "root_lower_spine",
    "upper_torso_arms",
    "right_leg",
    "upper_spine_neck",
    "left_leg",
    "head",
]


T2M_JOINT_NAMES = [
    "root/pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


KIT_JOINT_NAMES = [
    "root/pelvis",
    "spine1",
    "spine2",
    "neck",
    "head",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "right_toe",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "left_toe",
]


@dataclass(frozen=True)
class PartitionSpec:
    dataset: str
    feature_dim: int
    joints_num: int
    joint_names: list[str]
    part_joints_base: list[list[int]]
    overlap_pairs: list[tuple[int, int]]
    left_leg_part: int = 4
    right_leg_part: int = 2


def t2m_spec() -> PartitionSpec:
    return PartitionSpec(
        dataset="t2m",
        feature_dim=263,
        joints_num=22,
        joint_names=T2M_JOINT_NAMES,
        part_joints_base=[
            [0, 3],
            [6, 13, 14, 16, 17, 18, 19, 20, 21],
            [2, 5, 8, 11],
            [9, 12],
            [1, 4, 7, 10],
            [15],
        ],
        overlap_pairs=[
            (0, 1),
            (0, 2),
            (3, 1),
            (3, 2),
            (3, 6),
            (6, 1),
            (6, 2),
            (6, 12),
            (12, 6),
            (12, 15),
            (13, 12),
            (14, 12),
        ],
    )


def kit_spec() -> PartitionSpec:
    return PartitionSpec(
        dataset="kit",
        feature_dim=251,
        joints_num=21,
        joint_names=KIT_JOINT_NAMES,
        part_joints_base=[
            [0, 1],
            [2, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [3],
            [16, 17, 18, 19, 20],
            [4],
        ],
        overlap_pairs=[
            (0, 16),
            (0, 11),
            (1, 16),
            (1, 11),
            (1, 2),
            (2, 16),
            (2, 11),
            (2, 3),
            (3, 2),
            (3, 4),
            (5, 3),
            (8, 3),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["t2m", "kit"], required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Compare core generated fields against an existing partition JSON.",
    )
    return parser.parse_args()


def joint_feature_dims(joint_idx: int, joints_num: int) -> list[int]:
    pos_start = 4
    rot_start = pos_start + (joints_num - 1) * 3
    vel_start = rot_start + (joints_num - 1) * 6
    dims: list[int] = []
    if joint_idx == 0:
        dims.extend(range(0, 4))
    else:
        pos = pos_start + (joint_idx - 1) * 3
        rot = rot_start + (joint_idx - 1) * 6
        dims.extend(range(pos, pos + 3))
        dims.extend(range(rot, rot + 6))
    vel = vel_start + joint_idx * 3
    dims.extend(range(vel, vel + 3))
    return dims


def contact_dims(joints_num: int) -> tuple[list[int], list[int]]:
    start = 4 + (joints_num - 1) * 3 + (joints_num - 1) * 6 + joints_num * 3
    return [start, start + 1], [start + 2, start + 3]


def sorted_unique(values: Iterable[int]) -> list[int]:
    return sorted(set(int(v) for v in values))


def joint_to_part(part_joints_base: list[list[int]]) -> dict[int, int]:
    labels: dict[int, int] = {}
    for part_idx, joints in enumerate(part_joints_base):
        for joint_idx in joints:
            if joint_idx in labels:
                raise ValueError(f"Joint {joint_idx} is assigned to multiple base parts")
            labels[int(joint_idx)] = part_idx
    return labels


def build_partition(spec: PartitionSpec) -> dict:
    labels_by_joint = joint_to_part(spec.part_joints_base)
    if sorted(labels_by_joint) != list(range(spec.joints_num)):
        missing = sorted(set(range(spec.joints_num)) - set(labels_by_joint))
        raise ValueError(f"Base partition does not cover joints: {missing}")

    left_contacts, right_contacts = contact_dims(spec.joints_num)
    part_seg_no_overlap: list[list[int]] = []
    for part_idx, joints in enumerate(spec.part_joints_base):
        dims: list[int] = []
        for joint_idx in joints:
            dims.extend(joint_feature_dims(joint_idx, spec.joints_num))
        if part_idx == spec.left_leg_part:
            dims.extend(left_contacts)
        if part_idx == spec.right_leg_part:
            dims.extend(right_contacts)
        part_seg_no_overlap.append(sorted_unique(dims))

    part_joints_overlap_sources: list[list[int]] = [[] for _ in spec.part_joints_base]
    overlap_events = []
    for source_joint, target_joint in spec.overlap_pairs:
        source_part = labels_by_joint[source_joint]
        target_part = labels_by_joint[target_joint]
        if source_joint not in part_joints_overlap_sources[target_part]:
            part_joints_overlap_sources[target_part].append(source_joint)
        overlap_events.append(
            {
                "source_joint": source_joint,
                "source_joint_name": spec.joint_names[source_joint],
                "source_part": source_part,
                "target_joint": target_joint,
                "target_joint_name": spec.joint_names[target_joint],
                "target_part": target_part,
                "added_dims": len(joint_feature_dims(source_joint, spec.joints_num)),
            }
        )

    part_seg: list[list[int]] = []
    for part_idx, base_dims in enumerate(part_seg_no_overlap):
        dims = list(base_dims)
        for source_joint in part_joints_overlap_sources[part_idx]:
            dims.extend(joint_feature_dims(source_joint, spec.joints_num))
        part_seg.append(sorted_unique(dims))

    part_joints_effective = [
        sorted_unique(list(base) + list(overlap))
        for base, overlap in zip(spec.part_joints_base, part_joints_overlap_sources)
    ]
    joint_labels = [labels_by_joint[joint_idx] for joint_idx in range(spec.joints_num)]

    data = {
        "name": f"{spec.dataset}_skeleton_partition_formal_overlap",
        "dataset": spec.dataset,
        "feature_dim": spec.feature_dim,
        "joints_num": spec.joints_num,
        "n_parts": len(spec.part_joints_base),
        "part_names": PART_NAMES,
        "joint_names": spec.joint_names,
        "notes": (
            "Formal six-part overlap partition matching the released HumanML3D "
            "overlap-top3 tokenizer part order and boundary overlap rule."
        ),
        "partSeg": part_seg,
        "partSeg_no_overlap": part_seg_no_overlap,
        "joint_labels": joint_labels,
        "cluster_method": "average",
        "feature_mode": "relative_parent",
        "max_lag": 4,
        "parent_delta_rot": False,
        "parent_delta_vel": False,
        "enforce_chain_constraints": True,
        "chain_target_mode": "majority",
        "add_contact_part": False,
        "overlap_preset": "boundary_v1",
        "overlap_events": overlap_events,
        "part_joints_base": spec.part_joints_base,
        "part_joints_overlap_sources": part_joints_overlap_sources,
        "part_joints_effective": part_joints_effective,
        "source": {
            "generator": "tools/build_formal_skeleton_partition.py",
            "reference": "HumanML3D overlap-top3 skeleton_partition.json",
            "mapping": "same six-part order and boundary overlap rule; KIT uses its 21-joint skeleton indices",
        },
    }
    validate_partition(data)
    return data


def validate_partition(data: dict) -> None:
    feature_dim = int(data["feature_dim"])
    part_seg = data["partSeg"]
    part_seg_no_overlap = data["partSeg_no_overlap"]
    if len(part_seg) != int(data["n_parts"]):
        raise ValueError("partSeg length does not match n_parts")
    if len(part_seg_no_overlap) != int(data["n_parts"]):
        raise ValueError("partSeg_no_overlap length does not match n_parts")
    for key, segments in (("partSeg", part_seg), ("partSeg_no_overlap", part_seg_no_overlap)):
        for idx, segment in enumerate(segments):
            if len(segment) != len(set(segment)):
                raise ValueError(f"{key}[{idx}] contains duplicate dims")
            if any(dim < 0 or dim >= feature_dim for dim in segment):
                raise ValueError(f"{key}[{idx}] contains dims outside [0, {feature_dim})")

    no_overlap_counts = [0] * feature_dim
    for segment in part_seg_no_overlap:
        for dim in segment:
            no_overlap_counts[dim] += 1
    bad = [dim for dim, count in enumerate(no_overlap_counts) if count != 1]
    if bad:
        raise ValueError(f"partSeg_no_overlap does not cover feature dims exactly once: {bad[:16]}")


def compare_core(generated: dict, reference_path: Path) -> bool:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    keys = [
        "partSeg",
        "partSeg_no_overlap",
        "joint_labels",
        "cluster_method",
        "feature_mode",
        "max_lag",
        "parent_delta_rot",
        "parent_delta_vel",
        "enforce_chain_constraints",
        "chain_target_mode",
        "add_contact_part",
        "overlap_preset",
        "overlap_events",
        "part_joints_base",
        "part_joints_overlap_sources",
        "part_joints_effective",
    ]
    ok = True
    for key in keys:
        if generated.get(key) != reference.get(key):
            ok = False
            print(f"[compare] mismatch: {key}", file=sys.stderr)
    if ok:
        print(f"[compare] core fields match {reference_path}")
    return ok


def main() -> None:
    args = parse_args()
    spec = t2m_spec() if args.dataset == "t2m" else kit_spec()
    data = build_partition(spec)

    if args.compare is not None and not compare_core(data, args.compare):
        raise SystemExit(1)

    text = json.dumps(data, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[write] {args.output}")
        print(f"[summary] partSeg lengths: {[len(x) for x in data['partSeg']]}")
        print(f"[summary] no-overlap lengths: {[len(x) for x in data['partSeg_no_overlap']]}")


if __name__ == "__main__":
    main()
