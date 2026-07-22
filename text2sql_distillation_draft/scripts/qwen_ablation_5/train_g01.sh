#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GRID_ID="g1"
export SYNID_ALPHA="${SYNID_ALPHA:-0.1}"
export SYNID_BETA="${SYNID_BETA:-0.1}"

echo "Qwen ablation 5 grid g1"
echo "  alpha: ${SYNID_ALPHA}"
echo "  beta: ${SYNID_BETA}"

bash "${SCRIPT_DIR}/train_synid_sql_conventional.sh" "$@"
