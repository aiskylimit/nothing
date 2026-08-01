#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-teacher_lora_llama3_8b}"
export MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export CKPT_NAME="${CKPT_NAME:-llama3.1-8b-instruct}"
export MODEL_TYPE="${MODEL_TYPE:-llama}"
export DATA_DIR="${DATA_DIR:-processed_data/spider_data/llama}"
export USE_LORA="${USE_LORA:-1}"

exec bash "${SCRIPT_DIR}/../common/train_lm.sh" "$@"
