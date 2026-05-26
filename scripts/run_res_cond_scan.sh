#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "usage: $0 <gpu_id> <repeat_times> <base_name> <res_name> <res_cond_scale> [<res_cond_scale> ...]" >&2
  exit 1
fi

GPU_ID="$1"
REPEAT_TIMES="$2"
BASE_NAME="$3"
RES_NAME="$4"
shift 4

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

for SCALE in "$@"; do
  EXT="oldbase_newres_rsc${SCALE}_rt${REPEAT_TIMES}_eval_2026-03-08"
  LOG_FILE="$LOG_DIR/scan_oldbase_newres_rsc${SCALE}_rt${REPEAT_TIMES}_2026-03-08.log"
  echo "[scan] gpu=$GPU_ID res_cond_scale=$SCALE repeat_times=$REPEAT_TIMES log=$LOG_FILE"
  env CUDA_VISIBLE_DEVICES="$GPU_ID" python eval_t2m_trans_res.py \
    --dataset_name t2m \
    --name "$BASE_NAME" \
    --res_name "$RES_NAME" \
    --gpu_id 0 \
    --which_epoch net_best_fid.tar \
    --cond_scale 4 \
    --res_cond_scale "$SCALE" \
    --time_steps 10 \
    --temperature 1 \
    --topkr 0.9 \
    --repeat_times "$REPEAT_TIMES" \
    --ext "$EXT" > "$LOG_FILE" 2>&1
done
