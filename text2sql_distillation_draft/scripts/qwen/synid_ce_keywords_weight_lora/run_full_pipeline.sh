#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

echo "[pipeline] train + multi-seed infer"
bash scripts/qwen/synid_ce_keywords_weight_lora/run_full_grid_multiseed.sh "$@"

if [[ "${RUN_EVAL_AFTER_INFER:-false}" == "true" ]]; then
  echo "[pipeline] format + eval"
  bash scripts/qwen/synid_ce_keywords_weight_lora/format_eval_multiseed.sh
else
  echo "[pipeline] skip format + eval after infer"
fi

echo "[pipeline-done]"
