#! /usr/bin/env bash

set -euo pipefail

BASE_PATH="${BASE_PATH:-.}"
RUN_NAME="${RUN_NAME:?RUN_NAME must be set by the wrapper script}"
INFER_ROOT="${INFER_ROOT:-${BASE_PATH}/results/infer/${RUN_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-${BASE_PATH}/results/eval/${RUN_NAME}}"
ETYPE="${ETYPE:-all}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-60}"

BENCHMARKS=(
  "spider_data:spider_test"
  "spider_syn:spider_syn_test"
  "spider_realistic:spider_realistic_test"
  "spider_dk:spider_dk_test"
)

export PYTHONPATH="${BASE_PATH}"

for item in "${BENCHMARKS[@]}"; do
  benchmark="${item%%:*}"
  eval_key="${item##*:}"
  input_dir="${INFER_ROOT}/${benchmark}"
  if [[ ! -d "${input_dir}" ]]; then
    echo "[skip] missing inference dir: ${input_dir}" >&2
    continue
  fi

  for seed_dir in "${input_dir}"/seed*; do
    [[ -d "${seed_dir}" ]] || continue
    formatted_dir="${seed_dir}/formatted_data"
    eval_dir="${EVAL_ROOT}/${benchmark}/$(basename "${seed_dir}")"
    mkdir -p "${formatted_dir}" "${eval_dir}"

    python "${BASE_PATH}/scripts/format_spider_infer_results.py" \
      --input-dir "${seed_dir}" \
      --output-dir "${formatted_dir}"

    for pred in "${formatted_dir}"/*.pred.sql; do
      [[ -f "${pred}" ]] || continue
      gold="${pred%.pred.sql}.gold.sql"
      [[ -f "${gold}" ]] || continue
      log_path="${eval_dir}/$(basename "${pred%.pred.sql}").etype-${ETYPE}.timeout-${EXEC_TIMEOUT}.log"
      python "${BASE_PATH}/src/evaluator/run_benchmark.py" \
        --benchmark "${eval_key}" \
        --gold "${gold}" \
        --pred "${pred}" \
        --etype "${ETYPE}" \
        --exec_timeout "${EXEC_TIMEOUT}" \
        --progress_bar_for_each_datapoint \
        2>&1 | tee "${log_path}"
    done
  done
done
