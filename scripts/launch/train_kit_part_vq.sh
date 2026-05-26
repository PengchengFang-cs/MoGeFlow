#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

DATA_ROOT="${DATA_ROOT:-/path/to/KIT-ML}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/vqvae_kit_pscf}"

python tools/train_kv_part_vq.py \
  --dataset_name kit \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --partition_file configs/kit_skeleton_partition_pscf.json \
  --batch_size 256 \
  --window_size 64 \
  --total_iter 300000 \
  --warm_up_iter 1000 \
  --lr 2e-4 \
  --milestones 200000 \
  --gamma 0.05 \
  --weight_decay 0 \
  --commit 0.02 \
  --loss_vel 0.5 \
  --recons_loss l1_smooth \
  --code_dim 128 \
  --nb_code 128 \
  --mu 0.99 \
  --down_t 2 \
  --stride_t 2 \
  --width 512 \
  --depth 3 \
  --dilation_growth_rate 3 \
  --output_emb_width 128 \
  --vq_act relu \
  --quantizer ema_reset \
  --feat_bias 5 \
  --stat_bias_passes 2 \
  --print_iter 200 \
  --eval_iter 5000 \
  --eval_batch_size 32 \
  --save_latest 500 \
  --topk 3 \
  --num_workers 4 \
  --seed 3407 \
  --gpu_id 0
