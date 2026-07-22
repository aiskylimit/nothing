#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${UPLOAD_TO_HF:-true}" != "true" ]]; then
  echo "[upload-skip] UPLOAD_TO_HF=${UPLOAD_TO_HF:-}"
  exit 0
fi

if [[ -n "${HF_UPLOAD_TOKEN:-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}" ]]; then
  hf auth login --token "${HF_UPLOAD_TOKEN}"
fi

python ./scripts/upload_to_hf.py
