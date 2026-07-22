#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "[pipeline] qwen ablation 5 conventional-data SynID-SQL train + multi-seed infer"
bash scripts/qwen_ablation_5/run_all.sh "$@"

for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    echo "[pipeline] dry-run requested; skip format + eval"
    exit 0
  fi
done

ABLATION_SET="${ABLATION_SET:-g1,g2,g3}"
INFER_OUTPUT_ROOT_BASE="${INFER_OUTPUT_ROOT:-results/infer/qwen_ablation_5}"
EVAL_OUTPUT_ROOT_BASE="${EVAL_OUTPUT_ROOT:-results/eval/qwen_ablation_5}"

canonical_variant() {
  case "$1" in
    g1|g01)
      echo "g1"
      ;;
    g2|g02)
      echo "g2"
      ;;
    g3|g03|synid_sql|synid)
      echo "g3"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

IFS=', ' read -r -a VARIANTS <<< "${ABLATION_SET}"
for variant in "${VARIANTS[@]}"; do
  if [[ -z "${variant}" ]]; then
    continue
  fi

  variant_name="$(canonical_variant "${variant}")"
  echo "[pipeline] format + eval ${variant_name}"
  INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT_BASE}/${variant_name}/qwen_ablation_5" \
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT_BASE}/${variant_name}" \
    bash scripts/qwen_updated_2/synid_ce_keywords_weight_lora_218/format_eval_multiseed.sh
done

echo "[pipeline-done]"
