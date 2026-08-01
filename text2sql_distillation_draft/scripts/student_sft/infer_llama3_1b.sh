#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-student_sft_llama3_1b}"
export MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"

exec bash "${SCRIPT_DIR}/../common/infer_all.sh" "$@"
