#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-student_sft_qwen3_0.6b}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
export CKPT_NAME="${CKPT_NAME:-qwen3-0.6b}"
export MODEL_TYPE="${MODEL_TYPE:-qwen}"
export DATA_DIR="${DATA_DIR:-processed_data/benchmarks/spider_data/qwen}"
export USE_LORA="${USE_LORA:-0}"

exec bash "${SCRIPT_DIR}/../common/train_lm.sh" "$@"
