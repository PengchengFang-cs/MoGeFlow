#!/usr/bin/env bash
set -euo pipefail

REPO=/iridisfs/scratch/pf2m24/projects/Umdd/momask-codes
BLENDER=/scratch/pf2m24/tools/bin/blender
CHAR="${REPO}/figures/qualitative_20260608_retarget/assets/quaternius_rpg/RPG Characters - Nov 2020/Humanoid Rig Versions/Blends/Monk.blend"
RENDER="${REPO}/tools/render_quaternius_retarget_panel.py"

ROOT="${REPO}/figures/qual_selected_three_20260609_quaternius"
CELL_OUT="${ROOT}/cells_panel"
mkdir -p "${CELL_OUT}"

declare -A JOINTS=(
  [pscf_p0_crawl]="${REPO}/generation/qual_selected_three_20260609/ours_pscf_w200_bestfid_ema/joints/sample000_repeat00_len196_joints.npy"
  [pscf_p1_pickup]="${REPO}/generation/qual_selected_three_20260609/ours_pscf_w200_bestfid_ema/joints/sample001_repeat00_len196_joints.npy"
  [pscf_p2_kneel]="${REPO}/generation/qual_selected_three_20260609/ours_pscf_w200_bestfid_ema/joints/sample002_repeat00_len196_joints.npy"
  [mtransformer_p0_crawl]="${REPO}/generation/qual_selected_three_20260609/mtransformer/joints/sample0_repeat0_len196.npy"
  [mtransformer_p1_pickup]="${REPO}/generation/qual_selected_three_20260609/mtransformer/joints/sample1_repeat0_len196.npy"
  [mtransformer_p2_kneel]="${REPO}/generation/qual_selected_three_20260609/mtransformer/joints/sample2_repeat0_len196.npy"
  [momask_p0_crawl]="${REPO}/generation/qual_selected_three_20260609/momask/joints/sample000_repeat00_len196.npy"
  [momask_p1_pickup]="${REPO}/generation/qual_selected_three_20260609/momask/joints/sample001_repeat00_len196.npy"
  [momask_p2_kneel]="${REPO}/generation/qual_selected_three_20260609/momask/joints/sample002_repeat00_len196.npy"
)

for stem in \
  pscf_p0_crawl pscf_p1_pickup pscf_p2_kneel \
  mtransformer_p0_crawl mtransformer_p1_pickup mtransformer_p2_kneel \
  momask_p0_crawl momask_p1_pickup momask_p2_kneel
do
  echo "=== render ${stem} ==="
  "${BLENDER}" -b "${CHAR}" --python "${RENDER}" -- \
    --joints "${JOINTS[$stem]}" \
    --out "${CELL_OUT}/${stem}.png" \
    --frame-count 7 \
    --width 1200 \
    --height 720 \
    --align-route-x \
    --character-scale 0.74 \
    --lens 58 \
    --camera-side 0.72 \
    --camera-distance 2.10 \
    --camera-height 0.95
done

echo "Done. Quaternius cells: ${CELL_OUT}"
