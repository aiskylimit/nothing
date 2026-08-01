#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-teacher_lora_llama3_8b}"
export MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

exec bash "${SCRIPT_DIR}/../common/infer_all.sh" "$@"
