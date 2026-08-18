#!/usr/bin/env bash
set -euo pipefail

BASE_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash "${BASE_PATH}/scripts/math/train_pvsd_qwen3_4b.sh" \
    --save_steps 100 \
    --save_only_model true

CHECKPOINTS="${HOME}/outputs/checkpoints/pvsd/qwen3_4b/qwen3_4b_pvsd/checkpoint-500" \
  bash "${BASE_PATH}/scripts/math/eval_math.sh"
