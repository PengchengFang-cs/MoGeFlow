#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

PYTHON_BIN="${PYTHON_BIN:-/scratch/pf2m24/miniconda3/envs/fudoki-momask/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
RUN_NAME="${RUN_NAME:-codeflow_pscf_hml3d_clipqwen_codebook_h1152_p192_bs32x2}"
DATA_ROOT="${DATA_ROOT:-dataset/HumanML3D}"
KV_ROOT="${KV_ROOT:-/scratch/pf2m24/projects/Umdd/KV-Control}"
OUT_DIR="${OUT_DIR:-checkpoints/t2m/${RUN_NAME}}"

VQ_CHECKPOINT="${VQ_CHECKPOINT:-${KV_ROOT}/checkpoints/vqvae_overlap_top3_20260529_hf/new_vq_overlap_top3_20260529_best_top3.pth}"
VQ_PARTITION="${VQ_PARTITION:-${KV_ROOT}/checkpoints/vqvae_overlap_top3_20260529_hf/config/skeleton_partition.json}"
KV_PART_TARGET_MODE="${KV_PART_TARGET_MODE:-codebook}"
MEAN_PATH="${MEAN_PATH:-${KV_ROOT}/checkpoints/stats/mean.npy}"
STD_PATH="${STD_PATH:-${KV_ROOT}/checkpoints/stats/std.npy}"
QWEN_PATH="${QWEN_PATH:-/scratch/pf2m24/hf-models/Qwen3-8B}"
CLIP_HF_PATH="${CLIP_HF_PATH:-/scratch/pf2m24/hf-models/clip-vit-large-patch14}"
TEXT_CACHE_PATH="${TEXT_CACHE_PATH:-/scratch/pf2m24/text-caches/humanml3d_clip_l_qwen3_8b_hymotion_l128_v1}"

PART_HIDDEN_DIM="${PART_HIDDEN_DIM:-192}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1152}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEXT_REFINER_DEPTH="${TEXT_REFINER_DEPTH:-2}"

LAUNCH=("${PYTHON_BIN}")
if (( NPROC_PER_NODE > 1 )); then
  LAUNCH+=( -m torch.distributed.run --standalone --nproc_per_node "${NPROC_PER_NODE}" )
fi

"${LAUNCH[@]}" train_codeflow_part_structured.py \
  --name "${RUN_NAME}" \
  --output_dir "${OUT_DIR}" \
  --dataset_name t2m \
  --data_root "${DATA_ROOT}" \
  --kv_root "${KV_ROOT}" \
  --vq_backend kv_part \
  --vq_checkpoint "${VQ_CHECKPOINT}" \
  --vq_partition "${VQ_PARTITION}" \
  --kv_part_target_mode "${KV_PART_TARGET_MODE}" \
  --mean_path "${MEAN_PATH}" \
  --std_path "${STD_PATH}" \
  --text_encoder_type clip_qwen_cache \
  --text_cache_path "${TEXT_CACHE_PATH}" \
  --qwen_path "${QWEN_PATH}" \
  --clip_hf_path "${CLIP_HF_PATH}" \
  --qwen_max_length 128 \
  --text_refiner_depth "${TEXT_REFINER_DEPTH}" \
  --text_refiner_mlp_ratio 4.0 \
  --text_refiner_pool "${TEXT_REFINER_POOL:-all_tokens}" \
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
  --batch_size "${BATCH_SIZE}" \
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
\
  --best_checkpoint_limit 1 \
  --full_eval_every_epoch 10 \
  --full_eval_start_epoch 0 \
  --full_eval_batch_size 32 \
  --full_eval_num_workers 4 \
  --full_eval_steps 96 \
  --full_eval_cond_scale 6.0 \
  --full_eval_repeat_times 1 \
  --full_eval_seed 42 \
  --ddp_timeout_minutes 180 \
  "$@"
