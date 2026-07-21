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

run_qwen_updated_pipeline() {
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

run_qwen_updated_pipeline scripts/qwen_updated_2/synid_ce_keywords_weight_lora_218/run_full_pipeline.sh
run_qwen_updated_pipeline scripts/qwen_updated_2/synid_ce_keywords_weight_lora_436/run_full_pipeline.sh
run_qwen_updated_pipeline scripts/qwen_updated_2/synid_ce_no_keywords_weight_lora_218/run_full_pipeline.sh
run_qwen_updated_pipeline scripts/qwen_updated_2/synid_ce_no_keywords_weight_lora_436/run_full_pipeline.sh

bash scripts/qwen_updated_2/upload_to_hf.sh
