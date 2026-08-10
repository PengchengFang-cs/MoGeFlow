#!/usr/bin/env bash
set -euo pipefail

# Run this inside a Slurm GPU allocation.  The pinned dllm environment matches
# the Torch/Transformers versions recorded in the cache manifest, which is
# required for safe resume.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/pf2m24/miniconda3/envs/dllm/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset/HumanML3D}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/pf2m24/text-caches/humanml3d_qwen3_8b_normavg_userchat_l128_v1}"
QWEN_PATH="${QWEN_PATH:-/scratch/pf2m24/hf-models/Qwen3-8B}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-16}"

export TOKENIZERS_PARALLELISM=false
cd "${REPO_ROOT}"
PYTHONPATH=. "${PYTHON_BIN}" tools/precompute_qwen_normavg_text_cache.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --qwen_path "${QWEN_PATH}" \
  --qwen_max_length 128 \
  --layer_norm_eps 1e-5 \
  --batch_size "${BATCH_SIZE}" \
  --tokenizer_batch_size 256 \
  --device "${DEVICE}" \
  --model_dtype bfloat16 \
  --qwen_storage_dtype bfloat16 \
  --attn_implementation sdpa \
  --resume \
  "$@"
