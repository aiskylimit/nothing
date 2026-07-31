#! /usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
RUNNING_EXTRA_ARGS=("$@")

uv sync
source .venv/bin/activate

# hf download Dream-AI-HUST/sql_benchmarks \
#   --repo-type dataset \
#   --local-dir .
# unzip -o benchmarks.zip
# unzip -o data.zip
# unzip -o orig_processed_data.zip

export RUNNER_GPU_LIST="${QWEN25_GPU_LIST:-${RUNNER_GPU_LIST:-0,1,2,3,4,5,6,7}}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
export RUN_MODE="${RUN_MODE:-parallel}"
export SKIP_EXISTING="${SKIP_EXISTING:-false}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACC="${GRAD_ACC:-4}"
export INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-200}"
export SKIP_HF_UPLOAD="${SKIP_HF_UPLOAD:-1}"

BASELINE_DATA_DIR="${BASELINE_DATA_DIR:-orig_processed_data/benchmarks/spider_data/qwen}"
LOG_ROOT="${LOG_ROOT:-run_logs/qwen2.5/$(date +%Y%m%d_%H%M%S)}"

require_indexed_data() {
  local data_dir="${1%/}"
  local missing=0
  local required_files=(
    train_0.bin
    train_0.idx
    valid_0.bin
    valid_0.idx
  )

  for file in "${required_files[@]}"; do
    if [[ ! -s "${data_dir}/${file}" ]]; then
      echo "[qwen2.5] missing baseline/SFT data file: ${data_dir}/${file}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" -ne 0 ]]; then
    echo "[qwen2.5] expected baseline/SFT indexed data from orig_processed_data.zip under: ${data_dir}" >&2
    echo "[qwen2.5] override with BASELINE_DATA_DIR=/path/to/qwen if the zip extracts elsewhere." >&2
    exit 2
  fi
}

require_indexed_data "${BASELINE_DATA_DIR}"

run_group() {
  local name="$1"
  local filter="$2"
  local data_dir="${3:-}"
  local log_dir="${LOG_ROOT}/${name}"

  echo "[qwen2.5] running ${name}"
  echo "[qwen2.5] filter=${filter}"
  if [[ -n "${data_dir}" ]]; then
    echo "[qwen2.5] data_dir=${data_dir}"
  fi
  echo "[qwen2.5] gpus=${RUNNER_GPU_LIST}, gpus_per_job=${GPUS_PER_JOB}, mode=${RUN_MODE}"
  echo "[qwen2.5] logs=${log_dir}"

  if [[ -n "${data_dir}" ]]; then
    DATA_DIR="${data_dir}" bash running.sh \
      --mode "${RUN_MODE}" \
      --gpus "${RUNNER_GPU_LIST}" \
      --gpus-per-job "${GPUS_PER_JOB}" \
      --filter "${filter}" \
      --log-dir "${log_dir}" \
      --skip-finalize \
      "${RUNNING_EXTRA_ARGS[@]}"
  else
    bash running.sh \
      --mode "${RUN_MODE}" \
      --gpus "${RUNNER_GPU_LIST}" \
      --gpus-per-job "${GPUS_PER_JOB}" \
      --filter "${filter}" \
      --log-dir "${log_dir}" \
      --skip-finalize \
      "${RUNNING_EXTRA_ARGS[@]}"
  fi
}

run_group "sft" "scripts/qwen2.5/sft/train_qwen2.5_0.5b_sft.sh" "${BASELINE_DATA_DIR}" &
run_group "csd" "scripts/qwen2.5/kd/csd/train_0.5b_4b.sh" "${BASELINE_DATA_DIR}" &
run_group "distillm" "scripts/qwen2.5/kd/distillm/train_0.5b_4b.sh" "${BASELINE_DATA_DIR}" &
run_group "synid_sql" "scripts/qwen2.5/synid_sql/synid_ce_keywords_weight_lora_218/train_g" &

wait
