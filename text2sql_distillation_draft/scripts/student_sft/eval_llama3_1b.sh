#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-student_sft_llama3_1b}"

exec bash "${SCRIPT_DIR}/../common/eval_all.sh" "$@"
