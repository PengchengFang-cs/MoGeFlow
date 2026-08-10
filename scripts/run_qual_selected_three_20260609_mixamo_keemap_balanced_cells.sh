#!/usr/bin/env bash
set -euo pipefail

REPO=/iridisfs/scratch/pf2m24/projects/Umdd/momask-codes
BLENDER=/scratch/pf2m24/tools/bin/blender
CHAR="${REPO}/Ch46_nonPBR.fbx"
RENDER="${REPO}/tools/render_mixamo_keemap_bvh_panel.py"
MAPPING="${REPO}/assets/mapping.json"
KEEMAP_ROOT=/tmp/Keemap-Blender-Rig-ReTargeting-Addon

if [[ ! -d "${KEEMAP_ROOT}/Source/SourceFiles" ]]; then
  git clone --depth 1 https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon.git "${KEEMAP_ROOT}"
fi

ROOT="${ROOT_OVERRIDE:-${REPO}/figures/qual_selected_three_20260609_mixamo_balanced}"
CELL_OUT="${ROOT}/cells_panel"
DISPLAY_ALIGN="${DISPLAY_ALIGN:-raw}"
DISPLAY_PERP_SCALE="${DISPLAY_PERP_SCALE:-0.35}"
MATERIAL_LIGHTEN="${MATERIAL_LIGHTEN:-0.12}"
MATERIAL_DESATURATE="${MATERIAL_DESATURATE:-0.05}"
GROUND_GRAY="${GROUND_GRAY:-0.94}"
WORLD_GRAY="${WORLD_GRAY:-0.995}"
mkdir -p "${CELL_OUT}"

declare -A BVH=(
  [pscf_p0_crawl]="${REPO}/figures/qual_selected_three_20260609_bvh/pscf/pscf_p0_crawl.bvh"
  [pscf_p1_pickup]="${REPO}/figures/qual_selected_three_20260609_bvh/pscf/pscf_p1_pickup.bvh"
  [pscf_p2_kneel]="${REPO}/figures/qual_selected_three_20260609_bvh/pscf/pscf_p2_kneel.bvh"
  [mtransformer_p0_crawl]="${REPO}/figures/qual_selected_three_20260609_bvh/mtransformer/mtransformer_p0_crawl.bvh"
  [mtransformer_p1_pickup]="${REPO}/figures/qual_selected_three_20260609_bvh/mtransformer/mtransformer_p1_pickup.bvh"
  [mtransformer_p2_kneel]="${REPO}/figures/qual_selected_three_20260609_bvh/mtransformer/mtransformer_p2_kneel.bvh"
  [momask_p0_crawl]="${REPO}/figures/qual_selected_three_20260609_bvh/momask/momask_p0_crawl.bvh"
  [momask_p1_pickup]="${REPO}/figures/qual_selected_three_20260609_bvh/momask/momask_p1_pickup.bvh"
  [momask_p2_kneel]="${REPO}/figures/qual_selected_three_20260609_bvh/momask/momask_p2_kneel.bvh"
)

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
  if [[ -s "${CELL_OUT}/${stem}.png" ]]; then
    echo "=== skip ${stem} (exists) ==="
    continue
  fi
  echo "=== render ${stem} ==="
  "${BLENDER}" -b --python "${RENDER}" -- \
    --character-fbx "${CHAR}" \
    --bvh "${BVH[$stem]}" \
    --sample-joints "${JOINTS[$stem]}" \
    --sampling adaptive \
    --mapping "${MAPPING}" \
    --keemap-source "${KEEMAP_ROOT}/Source" \
    --out "${CELL_OUT}/${stem}.png" \
    --frame-count 6 \
    --start-frame 1 \
    --num-frames 196 \
    --width 1200 \
    --height 720 \
    --lens 58 \
    --camera-side 0.72 \
    --camera-distance 2.10 \
    --camera-height 0.95 \
    --display-scale 3.25 \
    --fallback-line-scale 2.85 \
    --display-align "${DISPLAY_ALIGN}" \
    --display-perp-scale "${DISPLAY_PERP_SCALE}" \
    --material-lighten "${MATERIAL_LIGHTEN}" \
    --material-desaturate "${MATERIAL_DESATURATE}" \
    --ground-gray "${GROUND_GRAY}" \
    --world-gray "${WORLD_GRAY}" \
    --route-z-ratio 0.025 \
    --route-width 0.008
done

echo "Done. Balanced Mixamo/KeeMap cells: ${CELL_OUT}"
