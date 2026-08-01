#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-teacher_lora_qwen3_4b}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export CKPT_NAME="${CKPT_NAME:-qwen3-4b}"
export MODEL_TYPE="${MODEL_TYPE:-qwen}"
export DATA_DIR="${DATA_DIR:-}"
export USE_LORA="${USE_LORA:-1}"

exec bash "${SCRIPT_DIR}/../common/train_lm.sh" "$@"
