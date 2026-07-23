#! /usr/bin/env bash

set -euo pipefail

DATA_DIR="${DATA_DIR:-orig_processed_data/benchmarks/spider_data/qwen}"
SAVE_TAG="${SAVE_TAG:-qwen_ablation_4_distillm}"
EPOCHS="${EPOCHS:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-4}"
OVERHEAD_MAX_STEPS="${OVERHEAD_MAX_STEPS:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"

export DATA_DIR
export SAVE_TAG
export EPOCHS
export EVAL_BATCH_SIZE
export BATCH_SIZE
export GRAD_ACC
export OVERHEAD_MAX_STEPS
export LOG_INTERVAL
export MAX_LENGTH
export MAX_PROMPT_LENGTH

echo "Qwen ablation 4 overhead: DistiLLM"
echo "  data dir: ${DATA_DIR}"
echo "  save tag: ${SAVE_TAG}"
echo "  epochs: ${EPOCHS}"
echo "  batch size: ${BATCH_SIZE}"
echo "  grad acc: ${GRAD_ACC}"
echo "  overhead max steps: ${OVERHEAD_MAX_STEPS}"
echo "  log interval: ${LOG_INTERVAL}"
echo "  eval batch size: ${EVAL_BATCH_SIZE}"
echo "  max length: ${MAX_LENGTH}"
echo "  max prompt length: ${MAX_PROMPT_LENGTH}"

bash scripts/kd_2/distillm/train_0.6b_4b.sh \
  --log-overhead-metrics \
  --overhead-method-name distillm \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACC}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-interval 0 \
  --log-interval "${LOG_INTERVAL}" \
  --overhead-max-steps "${OVERHEAD_MAX_STEPS}" \
  "$@"
