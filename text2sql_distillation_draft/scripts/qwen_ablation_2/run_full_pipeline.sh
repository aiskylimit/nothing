#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "[pipeline] qwen ablation 2 train + multi-seed infer"
bash scripts/qwen_ablation_2/run_all.sh "$@"

for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    echo "[pipeline] dry-run requested; skip format + eval"
    exit 0
  fi
done

ABLATION_SET="${ABLATION_SET:-no_syntax_weight}"
INFER_OUTPUT_ROOT_BASE="${INFER_OUTPUT_ROOT:-results/infer/qwen_ablation_2}"
EVAL_OUTPUT_ROOT_BASE="${EVAL_OUTPUT_ROOT:-results/eval/qwen_ablation_2}"

IFS=', ' read -r -a VARIANTS <<< "${ABLATION_SET}"
for variant in "${VARIANTS[@]}"; do
  if [[ -z "${variant}" ]]; then
    continue
  fi

  echo "[pipeline] format + eval no_syntax_weight"
  INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT_BASE}/no_syntax_weight" \
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT_BASE}/no_syntax_weight" \
    bash scripts/qwen_updated_2/synid_ce_keywords_weight_lora_218/format_eval_multiseed.sh
done

echo "[pipeline-done]"
