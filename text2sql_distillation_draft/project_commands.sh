#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

uv sync
source .venv/bin/activate

python -c "import nltk; nltk.download('punkt_tab')"

hf download Dream-AI-HUST/sql_benchmarks \
  --repo-type dataset \
  --local-dir .
unzip benchmarks.zip
unzip data.zip
# python ./scripts/synid_augment/build_teacher_train_from_final_merged.py

run_qwen_pipeline() {
  local pipeline_script="$1"

  RUNNER_GPU_LIST="${RUNNER_GPU_LIST:-0,1,2,3}" \
  GPUS_PER_JOB="${GPUS_PER_JOB:-4}" \
  RUN_MODE="${RUN_MODE:-parallel}" \
  SKIP_EXISTING="${SKIP_EXISTING:-false}" \
  INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}" \
  INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-128}" \
  bash "${pipeline_script}"
}

run_qwen_pipeline scripts/qwen/synid_ce_keywords_weight/run_full_pipeline.sh
run_qwen_pipeline scripts/qwen/synid_ce_no_keywords_weight/run_full_pipeline.sh
run_qwen_pipeline scripts/qwen/synid_ce_keywords_weight_lora/run_full_pipeline.sh
run_qwen_pipeline scripts/qwen/synid_ce_no_keywords_weight_lora/run_full_pipeline.sh

if [[ "${UPLOAD_TO_HF:-true}" == "true" ]]; then
  HF_UPLOAD_TOKEN="${HF_UPLOAD_TOKEN:-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}"
  if [[ -z "${HF_UPLOAD_TOKEN}" ]]; then
    echo "Set HF_UPLOAD_TOKEN before upload." >&2
    exit 1
  fi
  hf auth login --token "${HF_UPLOAD_TOKEN}"
  python ./scripts/upload_to_hf.py
fi
