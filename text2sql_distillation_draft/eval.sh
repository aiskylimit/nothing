#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 && "$1" != -* ]]; then
  export RUN_NAME="$1"
  shift
fi

export BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
export RUN_NAME="${RUN_NAME:-synid_sql_qwen3_4b_to_qwen3_0.6b}"
export INFER_ROOT="${INFER_ROOT:-${BASE_PATH}/results/infer/${RUN_NAME}}"
export EVAL_ROOT="${EVAL_ROOT:-${BASE_PATH}/results/eval/${RUN_NAME}}"

if [[ ! -d "${INFER_ROOT}" ]]; then
  echo "Missing inference results: ${INFER_ROOT}" >&2
  echo "Run inference first, or set RUN_NAME/INFER_ROOT to an existing result directory." >&2
  exit 1
fi

exec bash "${SCRIPT_DIR}/scripts/common/eval_all.sh" "$@"
