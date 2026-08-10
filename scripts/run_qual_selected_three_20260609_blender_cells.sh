#!/usr/bin/env bash
set -euo pipefail

module load ffmpeg

REPO=/iridisfs/scratch/pf2m24/projects/Umdd/momask-codes
BLENDER=/scratch/pf2m24/tools/bin/blender
RENDER_SCRIPT=/iridisfs/scratch/pf2m24/projects/Umdd/momask-codes-unifiedflow-base-trange-kvcontext/drawing_results/assets_plan/render_smpl_sequence_panel.py

ROOT="${REPO}/figures/qual_selected_three_20260609_smpl"
FIT_OUT="${ROOT}/smpl_fit_iter08"
FIT_INPUTS="${ROOT}/fit_inputs"
CELL_OUT="${ROOT}/cells_panel"

mkdir -p "${CELL_OUT}"

for stem in \
  pscf_p0_crawl pscf_p1_pickup pscf_p2_kneel \
  mtransformer_p0_crawl mtransformer_p1_pickup mtransformer_p2_kneel \
  momask_p0_crawl momask_p1_pickup momask_p2_kneel
do
  echo "=== render ${stem} ==="
  "${BLENDER}" -b --python "${RENDER_SCRIPT}" -- \
    --ply-dir "${FIT_OUT}/${stem}" \
    --joints "${FIT_INPUTS}/${stem}.npy" \
    --out "${CELL_OUT}/${stem}.png" \
    --mode root \
    --width 1200 \
    --height 720 \
    --frame-count 7 \
    --lens 58 \
    --camera-distance 2.08 \
    --camera-height 0.92 \
    --camera-side 0.78
done

echo "Done. Blender cells: ${CELL_OUT}"
