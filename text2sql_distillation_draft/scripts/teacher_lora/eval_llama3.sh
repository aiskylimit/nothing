#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-teacher_lora_llama3_8b}"

exec bash "${SCRIPT_DIR}/../common/eval_all.sh" "$@"
