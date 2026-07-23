#! /usr/bin/env bash

set -euo pipefail

DATA_DIR="${DATA_DIR:-orig_processed_data/benchmarks/spider_data/qwen}"
SAVE_TAG="${SAVE_TAG:-qwen_ablation_4_csd}"
EPOCHS="${EPOCHS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"

export DATA_DIR
export SAVE_TAG
export EPOCHS
export EVAL_BATCH_SIZE
export MAX_LENGTH
export MAX_PROMPT_LENGTH

echo "Qwen ablation 4 overhead: CSD"
echo "  data dir: ${DATA_DIR}"
echo "  save tag: ${SAVE_TAG}"
echo "  epochs: ${EPOCHS}"
echo "  eval batch size: ${EVAL_BATCH_SIZE}"
echo "  max length: ${MAX_LENGTH}"
echo "  max prompt length: ${MAX_PROMPT_LENGTH}"

bash scripts/kd_2/csd/train_0.6b_4b.sh \
  --log-overhead-metrics \
  --overhead-method-name csd \
  --epochs "${EPOCHS}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-interval 0 \
  "$@"
