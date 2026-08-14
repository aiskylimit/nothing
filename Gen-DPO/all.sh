#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="${1:-${MODEL_PATH:-${MODEL_NAME:-}}}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: bash run_all.sh <MODEL_PATH_OR_ID>"
  exit 1
fi

export MODEL_NAME="${MODEL_PATH}"
export MODEL_SLUG="$(basename "${MODEL_PATH}")"

bash arc.sh
bash mmlu.sh
bash qa.sh
bash wino.sh
bash hellaswag.sh
bash gsm8k.sh
