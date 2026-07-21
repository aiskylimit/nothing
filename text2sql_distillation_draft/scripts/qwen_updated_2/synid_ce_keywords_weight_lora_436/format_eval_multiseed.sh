#! /usr/bin/env bash

set -euo pipefail

if [[ -n "${RUN_MASTER_PORT:-}" && "${ALLOW_RUNNING_SH_UTILITY:-0}" != "1" ]]; then
  echo "[skip] format_eval_multiseed.sh is a utility; use it directly, not as a running.sh job."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
FORMAT_SCRIPT="${FORMAT_SCRIPT:-scripts/format_spider_infer_results.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-src/evaluator/run_benchmark.py}"
COLLECT_SCRIPT="${COLLECT_SCRIPT:-scripts/qwen_updated_2/synid_ce_keywords_weight_lora_436/collect_eval_results.py}"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-results/infer/synid_ce_keywords_weight_lora_436/qwen}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/eval/synid_ce_keywords_weight_lora_436/qwen}"
ETYPE="${ETYPE:-all}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-60}"
PROGRESS_BAR="${PROGRESS_BAR:-1}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-false}"

BENCHMARKS=(
  "spider_data:spider_test"
  "spider_syn:spider_syn_test"
  "spider_realistic:spider_realistic_test"
  "spider_dk:spider_dk_test"
)

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage: bash scripts/qwen_updated_2/synid_ce_keywords_weight_lora_436/format_eval_multiseed.sh [benchmark ...]

Formats and evaluates multi-seed inference outputs under:
  results/infer/synid_ce_keywords_weight_lora_436/qwen/<benchmark>/seed*/

Benchmark args, if provided, must be one or more of:
  spider_data spider_syn spider_realistic spider_dk

Environment overrides:
  INFER_OUTPUT_ROOT     Default: results/infer/synid_ce_keywords_weight_lora_436/qwen
  EVAL_OUTPUT_ROOT      Default: results/eval/synid_ce_keywords_weight_lora_436/qwen
  ETYPE                 Default: all
  EXEC_TIMEOUT          Default: 60
  PROGRESS_BAR=0        Disable per-datapoint progress bar
  SKIP_EXISTING_EVAL=1  Skip eval logs that already exist and are non-empty
EOF
}

safe_name() {
  local raw="$1"
  raw="${raw//[^[:alnum:]._-]/_}"
  printf '%s' "${raw}"
}

run_eval_for_seed_dir() {
  local benchmark="$1"
  local eval_key="$2"
  local seed_dir="$3"
  local seed_name
  local formatted_dir
  local eval_dir
  local pred
  local gold
  local run_name
  local log_path
  local -a cmd

  seed_name="$(basename "${seed_dir}")"
  formatted_dir="${seed_dir}/formatted_data"
  eval_dir="${EVAL_OUTPUT_ROOT}/${benchmark}/${seed_name}"

  shopt -s nullglob
  local input_files=("${seed_dir}"/*_sql_result.json)
  shopt -u nullglob
  if [[ "${#input_files[@]}" -eq 0 ]]; then
    echo "[skip] No *_sql_result.json files in ${seed_dir}" >&2
    return 0
  fi

  mkdir -p "${formatted_dir}" "${eval_dir}"

  echo "[format] ${benchmark}/${seed_name}"
  "${PYTHON_BIN}" "${FORMAT_SCRIPT}" \
    --input-dir "${seed_dir}" \
    --output-dir "${formatted_dir}"

  shopt -s nullglob
  local pred_files=("${formatted_dir}"/*.pred.sql)
  shopt -u nullglob
  if [[ "${#pred_files[@]}" -eq 0 ]]; then
    echo "[skip] No formatted predictions in ${formatted_dir}" >&2
    return 0
  fi

  for pred in "${pred_files[@]}"; do
    gold="${pred%.pred.sql}.gold.sql"
    if [[ ! -f "${gold}" ]]; then
      echo "[skip] Missing gold file for ${pred}: ${gold}" >&2
      continue
    fi

    run_name="$(safe_name "$(basename "${pred%.pred.sql}")")"
    log_path="${eval_dir}/${run_name}.etype-${ETYPE}.timeout-${EXEC_TIMEOUT}.log"

    if [[ "${SKIP_EXISTING_EVAL}" =~ ^(1|true|yes|y)$ && -s "${log_path}" ]]; then
      echo "[skip-eval] ${log_path}"
      continue
    fi

    cmd=(
      "${PYTHON_BIN}" "${EVAL_SCRIPT}"
      --benchmark "${eval_key}"
      --gold "${gold}"
      --pred "${pred}"
      --etype "${ETYPE}"
      --exec_timeout "${EXEC_TIMEOUT}"
    )
    if [[ "${PROGRESS_BAR}" =~ ^(1|true|yes|y)$ ]]; then
      cmd+=(--progress_bar_for_each_datapoint)
    fi

    echo "[eval] ${benchmark}/${seed_name} :: ${run_name}"
    echo "       pred: ${pred}"
    echo "       gold: ${gold}"
    echo "       log : ${log_path}"
    {
      echo "${cmd[*]}"
      "${cmd[@]}"
    } 2>&1 | tee "${log_path}"
  done
}

run_eval_for_benchmark() {
  local benchmark="$1"
  local eval_key="$2"
  local input_dir="${INFER_OUTPUT_ROOT}/${benchmark}"

  if [[ ! -d "${input_dir}" ]]; then
    echo "[skip] Missing inference output dir: ${input_dir}" >&2
    return 0
  fi

  shopt -s nullglob
  local seed_dirs=("${input_dir}"/seed*)
  shopt -u nullglob
  if [[ "${#seed_dirs[@]}" -eq 0 ]]; then
    echo "[skip] No seed directories in ${input_dir}" >&2
    return 0
  fi

  for seed_dir in "${seed_dirs[@]}"; do
    if [[ -d "${seed_dir}" ]]; then
      run_eval_for_seed_dir "${benchmark}" "${eval_key}" "${seed_dir}"
    fi
  done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -gt 0 ]]; then
  FILTERED_BENCHMARKS=()
  for requested in "$@"; do
    matched=0
    for item in "${BENCHMARKS[@]}"; do
      if [[ "${item%%:*}" == "${requested}" ]]; then
        FILTERED_BENCHMARKS+=("${item}")
        matched=1
        break
      fi
    done
    if [[ "${matched}" -eq 0 ]]; then
      echo "[error] Unknown benchmark: ${requested}" >&2
      usage >&2
      exit 2
    fi
  done
  BENCHMARKS=("${FILTERED_BENCHMARKS[@]}")
fi

for item in "${BENCHMARKS[@]}"; do
  run_eval_for_benchmark "${item%%:*}" "${item##*:}"
done

echo "[collect] writing per-seed eval JSON summaries"
"${PYTHON_BIN}" "${COLLECT_SCRIPT}" \
  --infer-output-root "${INFER_OUTPUT_ROOT}" \
  --eval-output-root "${EVAL_OUTPUT_ROOT}"
