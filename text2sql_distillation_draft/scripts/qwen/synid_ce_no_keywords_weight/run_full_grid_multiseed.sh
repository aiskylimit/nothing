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
RUNNER_GPU_LIST="${RUNNER_GPU_LIST:-0,1}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
INFER_BENCHMARKS="${INFER_BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-100}"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-results/infer/synid_ce_no_keywords_weight}"
INFER_FLUSH_EVERY="${INFER_FLUSH_EVERY:-100}"
INFER_CHECKPOINT_METRIC="${INFER_CHECKPOINT_METRIC:-exact_match}"
SYNID_DATASET_NAME="${SYNID_DATASET_NAME:-synid_privileged}"
DATA_DIR="${DATA_DIR:-processed_data/benchmarks/spider_data/${SYNID_DATASET_NAME}/qwen}"

export INFER_SEEDS
export FORMAT_AFTER_INFER
export SKIP_EXISTING
export SYNID_DATASET_NAME
export DATA_DIR

echo "[dataset] ${SYNID_DATASET_NAME}: ${DATA_DIR}"

bash running.sh \
  --mode "${RUN_MODE}" \
  --gpus "${RUNNER_GPU_LIST}" \
  --gpus-per-job "${GPUS_PER_JOB}" \
  --skip-finalize \
  --filter scripts/qwen/synid_ce_no_keywords_weight/train_g \
  --infer-after-train \
  --infer-script scripts/qwen/synid_ce_no_keywords_weight/infer_multiseed.py \
  --infer-benchmarks "${INFER_BENCHMARKS}" \
  --infer-split test \
  --infer-db full \
  --infer-batch-size "${INFER_BATCH_SIZE}" \
  --infer-output-root "${INFER_OUTPUT_ROOT}" \
  --infer-checkpoint-metric "${INFER_CHECKPOINT_METRIC}" \
  --infer-extra-args "--flush-every ${INFER_FLUSH_EVERY}" \
  "$@"
