#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-teacher_lora_qwen3_4b}"

exec bash "${SCRIPT_DIR}/../common/eval_all.sh" "$@"
