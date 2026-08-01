#! /usr/bin/env bash

set -euo pipefail

BASE_PATH="${BASE_PATH:-.}"
RUN_NAME="${RUN_NAME:?RUN_NAME must be set by the wrapper script}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set by the wrapper script}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH must point to a trained checkpoint or adapter}"
OUT_ROOT="${OUT_ROOT:-${BASE_PATH}/results/infer/${RUN_NAME}}"
DEVICE="${DEVICE:-cuda}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"

BENCHMARKS=(
  "spider_data:test:856"
  "spider_syn:test:756"
  "spider_realistic:test:755"
  "spider_dk:test:663"
)

export PYTHONPATH="${BASE_PATH}"
export INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"

for item in "${BENCHMARKS[@]}"; do
  benchmark="${item%%:*}"
  rest="${item#*:}"
  split="${rest%%:*}"
  max_new="${rest##*:}"
  output_path="${OUT_ROOT}/${benchmark}/${RUN_NAME}_${benchmark}_${split}_sql_result.json"

  python "${BASE_PATH}/scripts/common/infer_multiseed.py" \
    --model "${MODEL_PATH}" \
    --ckpt_path "${CHECKPOINT_PATH}" \
    --benchmark "${benchmark}" \
    --split "${split}" \
    --db full \
    --device "${DEVICE}" \
    --max-length "${max_new}" \
    --batch-size "${INFER_BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --output_path "${output_path}" \
    "$@"
done
