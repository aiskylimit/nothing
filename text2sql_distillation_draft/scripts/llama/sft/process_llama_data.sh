#! /bin/bash

set -euo pipefail

BASE_PATH="${BASE_PATH:-.}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
RAW_DATA_DIR="${RAW_DATA_DIR:-benchmarks_2/spider_data/format_data}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-processed_data/spider_data}"
DATA_PROCESS_WORKERS="${DATA_PROCESS_WORKERS:-8}"

# Llama 1B tokenizer stats on benchmarks_2/spider_data/format_data/train.jsonl:
# prompt max = 1496, prompt + response max = 1621.
MAX_LENGTH="${MAX_LENGTH:-1664}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
T_MAX_LENGTH="${T_MAX_LENGTH:-2048}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-1800}"

export PYTHONPATH="${BASE_PATH}"

for SPLIT in train valid test; do
  python "${BASE_PATH}/process_data.py" \
    --model-path "${MODEL_PATH}" \
    --model-type llama \
    --split "${SPLIT}" \
    --data-dir "${RAW_DATA_DIR}" \
    --processed-data-dir "${PROCESSED_DATA_ROOT}" \
    --data-process-workers "${DATA_PROCESS_WORKERS}" \
    --max-length "${MAX_LENGTH}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --t-max-length "${T_MAX_LENGTH}" \
    --t-max-prompt-length "${T_MAX_PROMPT_LENGTH}"
done
