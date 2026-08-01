#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-synid_sql_qwen3_4b_to_qwen3_0.6b}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

exec bash "${SCRIPT_DIR}/../common/infer_all.sh" "$@"
