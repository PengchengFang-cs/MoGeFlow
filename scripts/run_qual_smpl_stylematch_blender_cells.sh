#!/usr/bin/env bash
# Render the 9 SMPL cells of the style-matched qualitative grid with Blender.
set -euo pipefail

REPO=/iridisfs/scratch/pf2m24/projects/Umdd/momask-codes
BLENDER=/scratch/pf2m24/tools/bin/blender
FIT="${REPO}/figures/qual_selected_three_20260609_smpl/smpl_fit_iter08"
GEN="${REPO}/generation/qual_selected_three_20260609"
OUT="${REPO}/figures/qual_selected_three_20260820_smpl_stylematch/cells_panel"
SAMPLES="${SAMPLES:-128}"
W="${W:-2000}"
H="${H:-1300}"

mkdir -p "${OUT}"

render () {  # row col slug stem joints arrow
  echo "=== r$1 c$2 $4 ==="
  "${BLENDER}" -b --python "${REPO}/tools/blender_qual_cell.py" -- \
    --ply-dir "${FIT}/$4" \
    --joints "$5" \
    --arrow-color "$6" \
    --out "${OUT}/row0$1_col0$2_$3.png" \
    --width "${W}" --height "${H}" --samples "${SAMPLES}" \
    2>&1 | tail -2
}

PSCF="${GEN}/ours_pscf_w200_bestfid_ema/joints"
MTR="${GEN}/mtransformer/joints"
MOM="${GEN}/momask/joints"

render 0 0 ours_ps_cf     pscf_p0_crawl          "${PSCF}/sample000_repeat00_len196_joints.npy" "#2ca25f"
render 0 1 m_transformer  mtransformer_p0_crawl  "${MTR}/sample0_repeat0_len196.npy"            "#525252"
render 0 2 momask         momask_p0_crawl        "${MOM}/sample000_repeat00_len196.npy"         "#c2410c"
render 1 0 ours_ps_cf     pscf_p1_pickup         "${PSCF}/sample001_repeat00_len196_joints.npy" "#2ca25f"
render 1 1 m_transformer  mtransformer_p1_pickup "${MTR}/sample1_repeat0_len196.npy"            "#525252"
render 1 2 momask         momask_p1_pickup       "${MOM}/sample001_repeat00_len196.npy"         "#c2410c"
render 2 0 ours_ps_cf     pscf_p2_kneel          "${PSCF}/sample002_repeat00_len196_joints.npy" "#2ca25f"
render 2 1 m_transformer  mtransformer_p2_kneel  "${MTR}/sample2_repeat0_len196.npy"            "#525252"
render 2 2 momask         momask_p2_kneel        "${MOM}/sample002_repeat00_len196.npy"         "#c2410c"

echo "Done -> ${OUT}"
