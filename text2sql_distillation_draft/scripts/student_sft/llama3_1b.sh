#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-student_sft_llama3_1b}"
export MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
export CKPT_NAME="${CKPT_NAME:-llama3.2-1b-instruct}"
export MODEL_TYPE="${MODEL_TYPE:-llama}"
export DATA_DIR="${DATA_DIR:-processed_data/spider_data/llama}"
export USE_LORA="${USE_LORA:-0}"

exec bash "${SCRIPT_DIR}/../common/train_lm.sh" "$@"
