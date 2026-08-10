#!/usr/bin/env python3
"""Export the selected qualitative HumanML3D joints to BVH files.

This prepares the same kind of BVH motion assets used by MoMask's Blender /
Mixamo retargeting workflow. Run from the repository root.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from visualization.joints2bvh import Joint2BVHConvertor

ROOT = REPO / "figures" / "qual_selected_three_20260609_bvh"

MOTIONS = {
    "pscf": {
        "p0_crawl": REPO / "generation/qual_selected_three_20260609/ours_pscf_w200_bestfid_ema/joints/sample000_repeat00_len196_joints.npy",
        "p1_pickup": REPO / "generation/qual_selected_three_20260609/ours_pscf_w200_bestfid_ema/joints/sample001_repeat00_len196_joints.npy",
        "p2_kneel": REPO / "generation/qual_selected_three_20260609/ours_pscf_w200_bestfid_ema/joints/sample002_repeat00_len196_joints.npy",
    },
    "mtransformer": {
        "p0_crawl": REPO / "generation/qual_selected_three_20260609/mtransformer/joints/sample0_repeat0_len196.npy",
        "p1_pickup": REPO / "generation/qual_selected_three_20260609/mtransformer/joints/sample1_repeat0_len196.npy",
        "p2_kneel": REPO / "generation/qual_selected_three_20260609/mtransformer/joints/sample2_repeat0_len196.npy",
    },
    "momask": {
        "p0_crawl": REPO / "generation/qual_selected_three_20260609/momask/joints/sample000_repeat00_len196.npy",
        "p1_pickup": REPO / "generation/qual_selected_three_20260609/momask/joints/sample001_repeat00_len196.npy",
        "p2_kneel": REPO / "generation/qual_selected_three_20260609/momask/joints/sample002_repeat00_len196.npy",
    },
}


def main() -> None:
    converter = Joint2BVHConvertor()
    for method, motions in MOTIONS.items():
        out_dir = ROOT / method
        out_dir.mkdir(parents=True, exist_ok=True)
        for motion_name, joints_path in motions.items():
            joints = np.load(joints_path).astype(np.float32)
            raw_path = out_dir / f"{method}_{motion_name}.bvh"
            ik_path = out_dir / f"{method}_{motion_name}_ik.bvh"
            print(f"Export {raw_path.relative_to(REPO)}")
            converter.convert(joints, filename=str(raw_path), iterations=100, foot_ik=False)
            print(f"Export {ik_path.relative_to(REPO)}")
            converter.convert(joints, filename=str(ik_path), iterations=100, foot_ik=True)


if __name__ == "__main__":
    main()
