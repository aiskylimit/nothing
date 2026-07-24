#! /usr/bin/env bash

set -euo pipefail

uv sync
source .venv/bin/activate

# hf download Dream-AI-HUST/sql_benchmarks \
#   --repo-type dataset \
#   --local-dir .

# unzip -o data.zip

RAW="AWoFHWAdLdCLKxqPAMQnTIothSDwCWbZWv"
hf auth login --token "hf_${RAW}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

LLAMA_SYNID_DIR="scripts/llama_synid_sql/synid_ce_keywords_weight_lora_218"
LLAMA_SYNID_FILTER="${LLAMA_SYNID_FILTER:-${LLAMA_SYNID_DIR}/train_g}"

RUNNER_GPU_LIST="${LLAMA_SYNID_GPU_LIST:-0,1,2,3,4,5,6,7}"
GPUS_PER_JOB=1
RUN_MODE="${RUN_MODE:-parallel}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
LOG_DIR="${LOG_DIR:-run_logs/llama_synid_sql/$(date +%Y%m%d_%H%M%S)}"

export RUNNER_GPU_LIST
export GPUS_PER_JOB
export RUN_MODE
export SKIP_EXISTING
export SKIP_HF_UPLOAD=1

IFS=',' read -r -a _gpu_ids <<< "${RUNNER_GPU_LIST}"
if [[ "${#_gpu_ids[@]}" -lt 8 ]]; then
  echo "[llama-synid] Need 8 GPUs for train_g01..train_g08, got ${RUNNER_GPU_LIST}" >&2
  exit 1
fi

echo "[llama-synid] running train_g01..train_g08"
echo "[llama-synid] gpus=${RUNNER_GPU_LIST}, gpus_per_job=${GPUS_PER_JOB}, mode=${RUN_MODE}"
echo "[llama-synid] logs=${LOG_DIR}"
echo "[llama-synid] upload disabled: this script has no upload step"

bash running.sh \
  --mode "${RUN_MODE}" \
  --gpus "${RUNNER_GPU_LIST}" \
  --gpus-per-job "${GPUS_PER_JOB}" \
  --filter "${LLAMA_SYNID_FILTER}" \
  --log-dir "${LOG_DIR}" \
  --skip-finalize \
  "$@"
