#!/usr/bin/env python3
"""Recompute the six-part KIT partition with part-aware-vqvae's algorithm."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


JOINTS_NUM = 21
FEAT_DIM = 251
JOINT_NAMES = [
    "root/pelvis", "spine1", "spine2", "neck", "head",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle", "right_foot", "right_toe",
    "left_hip", "left_knee", "left_ankle", "left_foot", "left_toe",
]
PARENT_MAP = {
    0: -1, 1: 0, 2: 1, 3: 2, 4: 3,
    5: 3, 6: 5, 7: 6, 8: 3, 9: 8, 10: 9,
    11: 0, 12: 11, 13: 12, 14: 13, 15: 14,
    16: 0, 17: 16, 18: 17, 19: 18, 20: 19,
}
CONSTRAINT_CHAINS = {
    "right_leg": [11, 12, 13, 14, 15],
    "left_leg": [16, 17, 18, 19, 20],
    "right_arm": [5, 6, 7],
    "left_arm": [8, 9, 10],
}
BOUNDARY_RULES = [
    (0, [16, 11]),
    (1, [16, 11, 2]),
    (2, [16, 11, 3]),
    (3, [2, 4]),
    (5, [3]),
    (8, [3]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=Path("dataset/KIT-ML"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_frames", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_lag", type=int, default=4)
    return parser.parse_args()


def get_joint_dims(joint_id: int) -> dict[str, list[int]]:
    pos_start = 4
    rot_start = pos_start + (JOINTS_NUM - 1) * 3
    vel_start = rot_start + (JOINTS_NUM - 1) * 6
    if joint_id == 0:
        return {
            "all": [0, 1, 2, 3, vel_start, vel_start + 1, vel_start + 2],
            "ric": [0, 1, 2, 3],
            "rot": [],
            "vel": [vel_start, vel_start + 1, vel_start + 2],
        }
    ric = list(range(pos_start + (joint_id - 1) * 3, pos_start + joint_id * 3))
    rot = list(range(rot_start + (joint_id - 1) * 6, rot_start + joint_id * 6))
    vel = list(range(vel_start + joint_id * 3, vel_start + (joint_id + 1) * 3))
    return {"all": ric + rot + vel, "ric": ric, "rot": rot, "vel": vel}


def load_activations(data_root: Path) -> tuple[np.ndarray, int]:
    joint_dims = {j: get_joint_dims(j) for j in range(JOINTS_NUM)}
    ids = [x.strip() for x in (data_root / "train.txt").read_text().splitlines() if x.strip()]
    all_acts: list[np.ndarray] = []
    valid = 0
    for name in ids:
        path = data_root / "new_joint_vecs" / f"{name}.npy"
        if not path.is_file():
            continue
        try:
            motion = np.load(path)
        except Exception:
            continue
        if motion.ndim != 2 or motion.shape[1] != FEAT_DIM or motion.shape[0] < 10:
            continue
        acts = np.empty((motion.shape[0], JOINTS_NUM), dtype=np.float32)
        for joint in range(JOINTS_NUM):
            dims = joint_dims[joint]
            if joint == 0:
                feat = motion[:, dims["all"]]
            else:
                parent = PARENT_MAP[joint]
                ric = motion[:, dims["ric"]]
                if parent > 0:
                    ric = ric - motion[:, joint_dims[parent]["ric"]]
                feat = np.concatenate([ric, motion[:, dims["rot"]], motion[:, dims["vel"]]], axis=1)
            acts[:, joint] = np.linalg.norm(feat, axis=1)
        all_acts.append(acts)
        valid += 1
    if not all_acts:
        raise RuntimeError(f"No valid KIT motions found under {data_root}")
    return np.concatenate(all_acts, axis=0), valid


def lagged_similarity(activation: np.ndarray, max_lag: int) -> np.ndarray:
    x = activation.astype(np.float32, copy=False)
    z = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-8)
    sim = np.eye(JOINTS_NUM, dtype=np.float32)
    for i in range(JOINTS_NUM):
        for j in range(i + 1, JOINTS_NUM):
            best = 0.0
            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    a, b = z[:-lag, i], z[lag:, j]
                elif lag < 0:
                    a, b = z[-lag:, i], z[:lag, j]
                else:
                    a, b = z[:, i], z[:, j]
                best = max(best, abs(float(np.mean(a * b))))
            sim[i, j] = sim[j, i] = best
    np.nan_to_num(sim, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(sim, 0.0, 1.0)


def enforce_chains(labels: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    out = labels.copy()
    changes = []
    for name, chain in CONSTRAINT_CHAINS.items():
        old = [int(out[j]) for j in chain]
        uniq, counts = np.unique(old, return_counts=True)
        candidates = [int(x) for x, count in zip(uniq, counts) if count == counts.max()]
        target = candidates[0] if len(candidates) == 1 else next(x for x in old if x in candidates)
        for joint in chain:
            out[joint] = target
        if len(set(old)) > 1:
            changes.append({"chain": name, "joints": chain, "old_labels": old, "new_label": target, "strategy": "majority"})
    return out, changes


def split_largest(labels: np.ndarray, similarity: np.ndarray, protected: set[int]) -> tuple[np.ndarray, bool]:
    uniq, counts = np.unique(labels, return_counts=True)
    for label in [int(x) for x in uniq[np.argsort(-counts)]]:
        candidates = np.array([j for j in np.where(labels == label)[0].tolist() if j not in protected], dtype=np.int32)
        if len(candidates) < 2:
            continue
        dist = 1.0 - similarity[np.ix_(candidates, candidates)]
        np.fill_diagonal(dist, 0.0)
        sub = fcluster(linkage(squareform(dist, checks=False), method="average"), 2, criterion="maxclust")
        out = labels.copy()
        new_label = int(labels.max()) + 1
        for idx, joint in enumerate(candidates):
            if sub[idx] == 2:
                out[joint] = new_label
        return out, True
    return labels, False


def ensure_cluster_count(labels: np.ndarray, similarity: np.ndarray, target: int = 6) -> np.ndarray:
    out = labels.copy()
    protected = {j for chain in CONSTRAINT_CHAINS.values() for j in chain}
    while len(np.unique(out)) < target:
        out, ok = split_largest(out, similarity, protected)
        if not ok:
            break
    while len(np.unique(out)) > target:
        uniq, counts = np.unique(out, return_counts=True)
        smallest = int(uniq[np.argmin(counts)])
        members = np.where(out == smallest)[0]
        targets = [int(x) for x in uniq if int(x) != smallest]
        best = max(targets, key=lambda x: float(similarity[np.ix_(members, np.where(out == x)[0])].mean()))
        out[members] = best
    return out


def build_segments(labels: np.ndarray) -> tuple[list[list[int]], list[list[int]], list[dict]]:
    dims = {j: get_joint_dims(j)["all"] for j in range(JOINTS_NUM)}
    by_part: dict[int, list[int]] = defaultdict(list)
    for joint in range(JOINTS_NUM):
        by_part[int(labels[joint])].extend(dims[joint])
    ordered = sorted(by_part)
    parts = [sorted(set(by_part[label])) for label in ordered]
    left_part = ordered.index(int(labels[20]))
    right_part = ordered.index(int(labels[15]))
    parts[left_part] = sorted(set(parts[left_part] + [247, 248]))
    parts[right_part] = sorted(set(parts[right_part] + [249, 250]))
    no_overlap = [list(x) for x in parts]
    events = []
    for source, targets in BOUNDARY_RULES:
        source_part = int(labels[source])
        for target in targets:
            target_part = int(labels[target])
            if target_part == source_part:
                continue
            before = len(parts[target_part])
            parts[target_part] = sorted(set(parts[target_part]).union(dims[source]))
            added = len(parts[target_part]) - before
            if added:
                events.append({
                    "source_joint": source, "source_joint_name": JOINT_NAMES[source],
                    "source_part": source_part, "target_joint": target,
                    "target_joint_name": JOINT_NAMES[target], "target_part": target_part,
                    "added_dims": added,
                })
    return parts, no_overlap, events


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    activation, n_samples = load_activations(data_root)
    n_frames_full = int(activation.shape[0])
    if len(activation) > args.max_frames:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(activation), size=args.max_frames, replace=False)
        idx.sort()
        activation = activation[idx]
    similarity = lagged_similarity(activation, args.max_lag)
    dist = 1.0 - similarity
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), 6, criterion="maxclust")
    labels, chain_fixes = enforce_chains(labels)
    labels = ensure_cluster_count(labels, similarity, 6)
    labels, _ = enforce_chains(labels)
    remap = {old: new for new, old in enumerate(sorted(np.unique(labels).tolist()))}
    labels = np.asarray([remap[int(x)] for x in labels], dtype=np.int32)
    part_seg, part_seg_no_overlap, overlap_events = build_segments(labels)

    base = [[] for _ in part_seg]
    for joint, label in enumerate(labels):
        base[int(label)].append(joint)
    overlap_sources = [set() for _ in part_seg]
    for event in overlap_events:
        overlap_sources[event["target_part"]].add(event["source_joint"])
    overlap_sources_list = [sorted(x) for x in overlap_sources]

    result = {
        "name": "kit_part_aware_data_driven_overlap",
        "dataset": "kit", "feature_dim": FEAT_DIM, "joints_num": JOINTS_NUM,
        "joint_names": JOINT_NAMES, "partSeg": part_seg,
        "partSeg_no_overlap": part_seg_no_overlap, "n_parts": len(part_seg),
        "joint_labels": labels.tolist(), "cluster_method": "average",
        "feature_mode": "relative_parent", "max_lag": args.max_lag,
        "parent_delta_rot": False, "parent_delta_vel": False,
        "enforce_chain_constraints": True, "chain_target_mode": "majority",
        "add_contact_part": False, "overlap_preset": "boundary_v1",
        "overlap_events": overlap_events, "part_joints_base": base,
        "part_joints_overlap_sources": overlap_sources_list,
        "part_joints_effective": [sorted(set(base[i]).union(overlap_sources[i])) for i in range(len(base))],
        "n_samples": n_samples, "n_frames_full": n_frames_full,
        "n_frames_used": int(len(activation)), "chain_fixes": chain_fixes,
        "partition_seed": args.seed,
        "source": {
            "algorithm": "part-aware-vqvae/partition_analysis/analyze_skeleton_partition.py adapted to KIT-ML 21-joint features",
            "reference_commit": "3c4477789ce509057da303f7be25c163329c35ea",
            "data_root": str(data_root),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[partition] output={args.output}")
    print(f"[partition] samples={n_samples} frames={n_frames_full} used={len(activation)}")
    for idx, joints in enumerate(base):
        print(f"[partition] part={idx} joints={joints} names={[JOINT_NAMES[j] for j in joints]} dims={len(part_seg[idx])}")


if __name__ == "__main__":
    main()
