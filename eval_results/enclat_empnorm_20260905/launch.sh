#!/usr/bin/env bash
# Retrain the S (w075) and B (w100) width variants with continuous-decode full evals,
# so the S/B/L ablation rows share one decode protocol. Mirrors the 2026-06-01
# width-scaling recipe exactly except --decode_mode continuous.
# Usage: retrain_width_continuous_20260728.sh {s|b}
set -euo pipefail

PRESET="${1:?usage: s or b}"  # 原写法 {s|b}} 的花括号会截断参数展开，尾部 } 混入 $1
_ignore="${1:-
#|b}}"
case "${PRESET}" in
  s) PART_HIDDEN_DIM=96;  HIDDEN_SIZE=576;  TAG=w075_p96_h576 ;;
  b) PART_HIDDEN_DIM=128; HIDDEN_SIZE=768;  TAG=w100_p128_h768 ;;
  *) echo "unknown preset: ${PRESET}" >&2; exit 1 ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-/scratch/pf2m24/miniconda3/envs/fudoki-momask/bin/python}"
KV_ROOT="${KV_ROOT:-/scratch/pf2m24/projects/Umdd/KV-Control}"
DATA_ROOT="${DATA_ROOT:-/scratch/pf2m24/data/HumanML3D/HumanML3D}"
RUN_NAME="${RUN_NAME:-pscf_enclat_empnorm_${TAG}_d6s12_drop005_b64_lr1e4_e600_eval10from0_seed42_20260904}"
OUT_DIR="checkpoints/t2m/${RUN_NAME}"

cd /iridisfs/scratch/pf2m24/projects/Umdd/momask-codes

"${PYTHON_BIN}" train_codeflow_part_structured.py \
  --name "${RUN_NAME}" \
  --output_dir "${OUT_DIR}" \
  --dataset_name t2m \
  --data_root "${DATA_ROOT}" \
  --kv_root "${KV_ROOT}" \
  --vq_backend kv_part \
  --vq_checkpoint "${KV_ROOT}/checkpoints/vqvae_overlap_top3_20260529_hf/new_vq_overlap_top3_20260529_best_top3.pth" \
  --vq_partition "${KV_ROOT}/checkpoints/vqvae_overlap_top3_20260529_hf/config/skeleton_partition.json" \
  --mean_path "${KV_ROOT}/checkpoints/stats/mean.npy" \
  --std_path "${KV_ROOT}/checkpoints/stats/std.npy" \
  --clip_path "${KV_ROOT}/checkpoints/clip/ViT-B-32.pt" \
  --representation part_structured \
  --coupling_mode frame_grouped \
  --code_dim 128 \
  --num_parts 6 \
  --num_codes 128 \
  --part_hidden_dim "${PART_HIDDEN_DIM}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --num_heads 12 \
  --depth_double 6 \
  --depth_single 12 \
  --mlp_ratio 4.0 \
  --dropout 0.05 \
  --batch_size 64 \
  --max_epoch 600 \
  --lr 0.0001 \
  --lr_scheduler half_cosine \
  --eta_min_ratio 0.01 \
  --warmup_steps 2000 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --amp \
  --amp_dtype bf16 \
  --seed 42 \
  --cond_drop_prob 0.1 \
  --disable_self_condition \
  --time_schedule uniform \
  --latent_norm_mode empirical \
  --kv_part_target_mode encoder \
  --terminal_mode tied_logits \
  --terminal_tau_mode codebook_nn \
\
  --best_checkpoint_limit 1 \
  --full_eval_every_epoch 10 \
  --full_eval_start_epoch 0 \
  --full_eval_batch_size 32 \
  --full_eval_num_workers 4 \
  --full_eval_steps 96 \
  --full_eval_cond_scale 6.0 \
  --full_eval_repeat_times 1 \
  --full_eval_seed 42
