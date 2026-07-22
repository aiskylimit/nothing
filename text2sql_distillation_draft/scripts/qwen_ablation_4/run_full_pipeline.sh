#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "[pipeline] qwen ablation 4 computational-overhead runs"
bash scripts/qwen_ablation_4/run_all.sh "$@"

for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    echo "[pipeline] dry-run requested; skip collect"
    exit 0
  fi
done

python scripts/qwen_ablation_4/collect_overhead_results.py

echo "[pipeline-done]"
