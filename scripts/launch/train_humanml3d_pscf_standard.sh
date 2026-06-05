#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RUN_NAME="${RUN_NAME:-codeflow_part_structured_pscf_hml3d_standard}"
DATA_ROOT="${DATA_ROOT:-dataset/HumanML3D}"
KV_ROOT="${KV_ROOT:-.}"
OUT_DIR="${OUT_DIR:-checkpoints/t2m/${RUN_NAME}}"

VQ_CHECKPOINT="${VQ_CHECKPOINT:-${KV_ROOT}/checkpoints/vqvae/net_best_top3.pth}"
VQ_PARTITION="${VQ_PARTITION:-${KV_ROOT}/checkpoints/vqvae/skeleton_partition.json}"
MEAN_PATH="${MEAN_PATH:-${KV_ROOT}/checkpoints/stats/mean.npy}"
STD_PATH="${STD_PATH:-${KV_ROOT}/checkpoints/stats/std.npy}"
CLIP_PATH="${CLIP_PATH:-${KV_ROOT}/checkpoints/clip/ViT-B-32.pt}"

PART_HIDDEN_DIM="${PART_HIDDEN_DIM:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"

python train_codeflow_part_structured.py \
  --name "${RUN_NAME}" \
  --output_dir "${OUT_DIR}" \
  --dataset_name t2m \
  --data_root "${DATA_ROOT}" \
  --kv_root "${KV_ROOT}" \
  --vq_backend kv_part \
  --vq_checkpoint "${VQ_CHECKPOINT}" \
  --vq_partition "${VQ_PARTITION}" \
  --mean_path "${MEAN_PATH}" \
  --std_path "${STD_PATH}" \
  --clip_path "${CLIP_PATH}" \
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
  --latent_norm_mode codebook \
  --terminal_mode tied_logits \
  --terminal_tau_mode codebook_nn \
  --terminal_loss_weight 0.0 \
  --clean_loss_weight 0.0 \
  --best_checkpoint_limit 3 \
  --full_eval_every_epoch 10 \
  --full_eval_start_epoch 0 \
  --full_eval_batch_size 32 \
  --full_eval_num_workers 4 \
  --full_eval_steps 96 \
  --full_eval_cond_scale 6.0 \
  --full_eval_repeat_times 1 \
  --full_eval_seed 42 \
  "$@"
