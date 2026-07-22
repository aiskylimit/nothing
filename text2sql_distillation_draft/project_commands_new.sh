#! /usr/bin/env bash

set -euo pipefail

uv sync
source .venv/bin/activate

# python -c "import nltk; nltk.download('punkt_tab')"

# hf download Dream-AI-HUST/sql_benchmarks \
#   --repo-type dataset \
#   --local-dir .
# unzip -o benchmarks.zip
# unzip -o data.zip
# python ./scripts/synid_augment/build_teacher_train_from_final_merged.py

COMMON_ENV=(
  RUNNER_GPU_LIST=4,5,6,7
  GPUS_PER_JOB=1
  RUN_MODE=parallel
  SKIP_EXISTING=false
  INFER_SEEDS=10,42,50,100,1234
  EVAL_BATCH_SIZE=32
  INFER_BATCH_SIZE=128
)

run_pipeline() {
  local script="$1"

  echo "[project] running ${script}"
  env "${COMMON_ENV[@]}" bash "${script}"
}

# Run qwen_updated_3 first.
run_pipeline scripts/qwen_updated_3/synid_ce_keywords_weight_lora/run_full_pipeline.sh
run_pipeline scripts/qwen_updated_3/synid_ce_no_keywords_weight_lora/run_full_pipeline.sh

run_pipeline scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/run_full_pipeline.sh
run_pipeline scripts/qwen_updated_3_218/synid_ce_no_keywords_weight_lora/run_full_pipeline.sh

run_pipeline scripts/qwen_updated_3_436/synid_ce_keywords_weight_lora/run_full_pipeline.sh
run_pipeline scripts/qwen_updated_3_436/synid_ce_no_keywords_weight_lora/run_full_pipeline.sh
