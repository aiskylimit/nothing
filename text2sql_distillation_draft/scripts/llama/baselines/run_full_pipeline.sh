#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

INFER_BENCHMARKS="${INFER_BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
INFER_OUTPUT_ROOT_BASE="${INFER_OUTPUT_ROOT_BASE:-results/infer/llama_baselines}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/eval/llama_baselines/llama}"

echo "[pipeline] train + multi-seed infer"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT_BASE}" \
INFER_BENCHMARKS="${INFER_BENCHMARKS}" \
  bash scripts/llama/baselines/run_full_grid_multiseed.sh "$@"

echo "[pipeline] format + eval"
IFS=',' read -r -a EVAL_BENCHMARK_ARGS <<< "${INFER_BENCHMARKS}"
for i in "${!EVAL_BENCHMARK_ARGS[@]}"; do
  EVAL_BENCHMARK_ARGS[$i]="${EVAL_BENCHMARK_ARGS[$i]//[[:space:]]/}"
done
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT_BASE}/llama" \
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
  bash scripts/llama/baselines/format_eval_multiseed.sh "${EVAL_BENCHMARK_ARGS[@]}"

case "${SKIP_HF_UPLOAD:-0}" in
  1|true|TRUE|yes|YES|y|Y)
    echo "[upload-skip] SKIP_HF_UPLOAD=${SKIP_HF_UPLOAD}"
    ;;
  *)
    echo "[pipeline] upload to Hugging Face"
    "${PYTHON:-python}" scripts/llama/baselines/upload_to_hf.py
    ;;
esac

echo "[pipeline-done]"
