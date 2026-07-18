#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TARGET="${TARGET:-all}" # all, 8b, or 1b
INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"
FORMAT_AFTER_INFER="${FORMAT_AFTER_INFER:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
RUN_MODE="${RUN_MODE:-sequential}"
INFER_BENCHMARKS="${INFER_BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"
INFER_OUTPUT_ROOT_BASE="${INFER_OUTPUT_ROOT_BASE:-results/infer/llama_sft}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/eval/llama_sft/llama}"
INFER_FLUSH_EVERY="${INFER_FLUSH_EVERY:-100}"
INFER_CHECKPOINT_METRIC="${INFER_CHECKPOINT_METRIC:-exact_match}"
RUNNER_GPU_LIST_8B="${RUNNER_GPU_LIST_8B:-0,1}"
RUNNER_GPU_LIST_1B="${RUNNER_GPU_LIST_1B:-0}"
GPUS_PER_JOB_8B="${GPUS_PER_JOB_8B:-2}"
GPUS_PER_JOB_1B="${GPUS_PER_JOB_1B:-1}"

export INFER_SEEDS
export FORMAT_AFTER_INFER
export SKIP_EXISTING

run_target() {
  local label="$1"
  local filter="$2"
  local gpu_list="$3"
  local gpus_per_job="$4"

  echo "[pipeline] train + infer ${label}"
  bash running.sh \
    --mode "${RUN_MODE}" \
    --gpus "${gpu_list}" \
    --gpus-per-job "${gpus_per_job}" \
    --skip-finalize \
    --filter "${filter}" \
    --infer-after-train \
    --infer-script scripts/qwen/synid_ce_multilayer_3/infer_multiseed.py \
    --infer-benchmarks "${INFER_BENCHMARKS}" \
    --infer-split test \
    --infer-db full \
    --infer-batch-size "${INFER_BATCH_SIZE}" \
    --infer-output-root "${INFER_OUTPUT_ROOT_BASE}" \
    --infer-checkpoint-metric "${INFER_CHECKPOINT_METRIC}" \
    --infer-extra-args "--flush-every ${INFER_FLUSH_EVERY}"
}

case "${TARGET}" in
  all)
    run_target "llama3 8b lora" "scripts/llama/sft/sft_llama3_8b_lora.sh" "${RUNNER_GPU_LIST_8B}" "${GPUS_PER_JOB_8B}"
    run_target "llama3 1b full sft" "scripts/llama/sft/sft_llama3_1b.sh" "${RUNNER_GPU_LIST_1B}" "${GPUS_PER_JOB_1B}"
    ;;
  8b)
    run_target "llama3 8b lora" "scripts/llama/sft/sft_llama3_8b_lora.sh" "${RUNNER_GPU_LIST_8B}" "${GPUS_PER_JOB_8B}"
    ;;
  1b)
    run_target "llama3 1b full sft" "scripts/llama/sft/sft_llama3_1b.sh" "${RUNNER_GPU_LIST_1B}" "${GPUS_PER_JOB_1B}"
    ;;
  *)
    echo "TARGET must be one of: all, 8b, 1b" >&2
    exit 2
    ;;
esac

echo "[pipeline] format + eval"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT_BASE}/llama" \
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
  bash scripts/llama/sft/format_eval_multiseed.sh
