#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

uv sync
source .venv/bin/activate

python -c "import nltk; nltk.download('punkt_tab')"

export GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
export RUN_MODE="${RUN_MODE:-sequential}"
export SKIP_EXISTING="${SKIP_EXISTING:-false}"
export INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-128}"

ABLATION_1_GPU_LIST="${ABLATION_1_GPU_LIST:-0}"
ABLATION_2_GPU_LIST="${ABLATION_2_GPU_LIST:-1}"
ABLATION_3_GPU_LIST="${ABLATION_3_GPU_LIST:-2}"

echo "[ablation] launching ablation 1 on GPU(s): ${ABLATION_1_GPU_LIST}"
RUNNER_GPU_LIST="${ABLATION_1_GPU_LIST}" bash scripts/qwen_ablation_1/run_full_pipeline.sh &
ABLATION_1_PID="$!"

echo "[ablation] launching ablation 2 on GPU(s): ${ABLATION_2_GPU_LIST}"
RUNNER_GPU_LIST="${ABLATION_2_GPU_LIST}" bash scripts/qwen_ablation_2/run_full_pipeline.sh &
ABLATION_2_PID="$!"

echo "[ablation] launching ablation 3 on GPU(s): ${ABLATION_3_GPU_LIST}"
RUNNER_GPU_LIST="${ABLATION_3_GPU_LIST}" bash scripts/qwen_ablation_3/run_full_pipeline.sh &
ABLATION_3_PID="$!"

STATUS=0
if ! wait "${ABLATION_1_PID}"; then
  echo "[ablation] ablation 1 failed" >&2
  STATUS=1
fi
if ! wait "${ABLATION_2_PID}"; then
  echo "[ablation] ablation 2 failed" >&2
  STATUS=1
fi
if ! wait "${ABLATION_3_PID}"; then
  echo "[ablation] ablation 3 failed" >&2
  STATUS=1
fi

if [[ "${STATUS}" -ne 0 ]]; then
  exit "${STATUS}"
fi

bash scripts/qwen_updated_2/upload_to_hf.sh
