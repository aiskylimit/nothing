#! /usr/bin/env bash
uv sync
source .venv/bin/activate

# python -c "import nltk; nltk.download('punkt_tab')"

hf download Dream-AI-HUST/sql_benchmarks \
  --repo-type dataset \
  --local-dir .
# unzip -o benchmarks.zip
# unzip -o data.zip
unzip -o orig_processed_data.zip
# python ./scripts/synid_augment/build_teacher_train_from_final_merged.py

RUNNER_GPU_LIST=1 GPUS_PER_JOB=1 bash scripts/qwen_ablation_5/run_full_pipeline.sh
