#!/usr/bin/env bash
set -euo pipefail

module load cuda/12.4.0
module load gcc/11.5.0
module load ffmpeg

REPO=/iridisfs/scratch/pf2m24/projects/Umdd/momask-codes
SMPL_ROOT=/scratch/pf2m24/projects/Umdd/momask-kvcontext-rl/third_party/HumanML3D-SMPL
PY=/scratch/pf2m24/miniconda3/envs/fudoki-momask/bin/python

FIT_INPUTS="${REPO}/figures/qual_selected_three_20260609_smpl/fit_inputs"
FIT_OUT="${REPO}/figures/qual_selected_three_20260609_smpl/smpl_fit_iter08"

mkdir -p "${FIT_OUT}"
export PYTHONPATH="${SMPL_ROOT}:${PYTHONPATH:-}"
cd "${SMPL_ROOT}"

for stem in \
  pscf_p0_crawl pscf_p1_pickup pscf_p2_kneel \
  mtransformer_p0_crawl mtransformer_p1_pickup mtransformer_p2_kneel \
  momask_p0_crawl momask_p1_pickup momask_p2_kneel
do
  echo "=== ${stem} ==="
  "${PY}" joints2smpl/fit_seq.py \
    --num_smplify_iters 8 \
    --cuda True \
    --gpu_ids 0 \
    --num_joints 22 \
    --joint_category AMASS \
    --data_folder "${FIT_INPUTS}" \
    --save_folder "${FIT_OUT}" \
    --files "${stem}.npy"
done

echo "Done. SMPLify output: ${FIT_OUT}"
