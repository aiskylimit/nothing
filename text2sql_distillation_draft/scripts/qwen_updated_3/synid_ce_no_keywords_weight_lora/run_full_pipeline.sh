#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

echo "[pipeline] train + multi-seed infer"
bash scripts/qwen_updated_3/synid_ce_no_keywords_weight_lora/run_full_grid_multiseed.sh "$@"

echo "[pipeline] format + eval"
bash scripts/qwen_updated_3/synid_ce_no_keywords_weight_lora/format_eval_multiseed.sh

echo "[pipeline-done]"
