#! /usr/bin/env bash

set -euo pipefail

if [[ -n "${RUN_MASTER_PORT:-}" && "${ALLOW_RUNNING_SH_UTILITY:-0}" != "1" ]]; then
  echo "[skip] run_full_grid_multiseed.sh is a wrapper; use it directly, not as a running.sh job."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"
FORMAT_AFTER_INFER="${FORMAT_AFTER_INFER:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
RUN_MODE="${RUN_MODE:-sequential}"
RUNNER_GPU_LIST="${RUNNER_GPU_LIST:-0}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
INFER_BENCHMARKS="${INFER_BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-100}"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-results/infer/synid_ce_keywords_weight_lora}"
INFER_FLUSH_EVERY="${INFER_FLUSH_EVERY:-100}"
INFER_CHECKPOINT_METRIC="${INFER_CHECKPOINT_METRIC:-exact_match}"

export INFER_SEEDS
export FORMAT_AFTER_INFER
export SKIP_EXISTING

TARGET_KD_GRID_FILTERS=(
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g01.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g02.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g03.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g04.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g05.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g06.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g07.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g08.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g09.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g10.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g11.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g12.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g13.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g14.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g15.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g16.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g17.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g18.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g19.sh
  scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/train_g20.sh
)

for grid_filter in "${TARGET_KD_GRID_FILTERS[@]}"; do
  bash running.sh \
    --mode "${RUN_MODE}" \
    --gpus "${RUNNER_GPU_LIST}" \
    --gpus-per-job "${GPUS_PER_JOB}" \
    --skip-finalize \
    --filter "${grid_filter}" \
    --infer-after-train \
    --infer-script scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/infer_multiseed.py \
    --infer-benchmarks "${INFER_BENCHMARKS}" \
    --infer-split test \
    --infer-db full \
    --infer-batch-size "${INFER_BATCH_SIZE}" \
    --infer-output-root "${INFER_OUTPUT_ROOT}" \
    --infer-checkpoint-metric "${INFER_CHECKPOINT_METRIC}" \
    --infer-extra-args "--flush-every ${INFER_FLUSH_EVERY}" \
    "$@"
done
