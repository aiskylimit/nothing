#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

uv sync
source .venv/bin/activate

python -c "import nltk; nltk.download('punkt_tab')"

RUNNER_GPU_LIST="${RUNNER_GPU_LIST:-0,1}" \
GPUS_PER_JOB="${GPUS_PER_JOB:-1}" \
RUN_MODE="${RUN_MODE:-sequential}" \
SKIP_EXISTING="${SKIP_EXISTING:-false}" \
INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}" \
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-128}" \
bash scripts/qwen_ablation_1/run_full_pipeline.sh

bash scripts/qwen_updated_2/upload_to_hf.sh
