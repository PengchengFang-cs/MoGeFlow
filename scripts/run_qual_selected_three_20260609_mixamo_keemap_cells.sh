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

ROOT="${REPO}/figures/qual_selected_three_20260609_mixamo"
CELL_OUT="${ROOT}/cells_panel"
mkdir -p "${CELL_OUT}"

declare -A BVH=(
  [pscf_p0_crawl]="${REPO}/figures/qual_selected_three_20260609_bvh/pscf/pscf_p0_crawl_ik.bvh"
  [pscf_p1_pickup]="${REPO}/figures/qual_selected_three_20260609_bvh/pscf/pscf_p1_pickup_ik.bvh"
  [pscf_p2_kneel]="${REPO}/figures/qual_selected_three_20260609_bvh/pscf/pscf_p2_kneel_ik.bvh"
  [mtransformer_p0_crawl]="${REPO}/figures/qual_selected_three_20260609_bvh/mtransformer/mtransformer_p0_crawl_ik.bvh"
  [mtransformer_p1_pickup]="${REPO}/figures/qual_selected_three_20260609_bvh/mtransformer/mtransformer_p1_pickup_ik.bvh"
  [mtransformer_p2_kneel]="${REPO}/figures/qual_selected_three_20260609_bvh/mtransformer/mtransformer_p2_kneel_ik.bvh"
  [momask_p0_crawl]="${REPO}/figures/qual_selected_three_20260609_bvh/momask/momask_p0_crawl_ik.bvh"
  [momask_p1_pickup]="${REPO}/figures/qual_selected_three_20260609_bvh/momask/momask_p1_pickup_ik.bvh"
  [momask_p2_kneel]="${REPO}/figures/qual_selected_three_20260609_bvh/momask/momask_p2_kneel_ik.bvh"
)

for stem in \
  pscf_p0_crawl pscf_p1_pickup pscf_p2_kneel \
  mtransformer_p0_crawl mtransformer_p1_pickup mtransformer_p2_kneel \
  momask_p0_crawl momask_p1_pickup momask_p2_kneel
do
  echo "=== render ${stem} ==="
  "${BLENDER}" -b --python "${RENDER}" -- \
    --character-fbx "${CHAR}" \
    --bvh "${BVH[$stem]}" \
    --mapping "${MAPPING}" \
    --keemap-source "${KEEMAP_ROOT}/Source" \
    --out "${CELL_OUT}/${stem}.png" \
    --frame-count 7 \
    --start-frame 1 \
    --num-frames 196 \
    --width 1200 \
    --height 720 \
    --lens 58 \
    --camera-side 0.72 \
    --camera-distance 2.10 \
    --camera-height 0.95
done

echo "Done. Mixamo/KeeMap cells: ${CELL_OUT}"
