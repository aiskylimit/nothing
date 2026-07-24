#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

TEACHER_VLLM_BASE_URL="${TEACHER_VLLM_BASE_URL:-http://localhost:8101/v1}"
STUDENT_VLLM_BASE_URL="${STUDENT_VLLM_BASE_URL:-http://localhost:8102/v1}"
STUDENT_SFT_VLLM_BASE_URL="${STUDENT_SFT_VLLM_BASE_URL:-${STUDENT_VLLM_BASE_URL}}"
CSD_VLLM_BASE_URL="${CSD_VLLM_BASE_URL:-http://localhost:8103/v1}"
DISTILLM_VLLM_BASE_URL="${DISTILLM_VLLM_BASE_URL:-http://localhost:8103/v1}"
SYNID_SQL_VLLM_BASE_URL="${SYNID_SQL_VLLM_BASE_URL:-http://localhost:8104/v1}"
TEACHER_VLLM_MODEL="${TEACHER_VLLM_MODEL:-teacher_lora}"
STUDENT_BASE_VLLM_MODEL="${STUDENT_BASE_VLLM_MODEL:-student_sft}"
STUDENT_SFT_VLLM_MODEL="${STUDENT_SFT_VLLM_MODEL:-student_sft}"
CSD_VLLM_MODEL="${CSD_VLLM_MODEL:-csd}"
DISTILLM_VLLM_MODEL="${DISTILLM_VLLM_MODEL:-distillm}"
SYNID_SQL_VLLM_MODEL="${SYNID_SQL_VLLM_MODEL:-synid_sql}"

BENCHMARKS="${BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
RUNS="${RUNS:-teacher_lora,student_sft,distillm,csd,synid_sql}"
SEEDS="${SEEDS:-42}"
SPLIT="${SPLIT:-test}"
DB="${DB:-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/macsql_vllm}"
PROMPT_DIR="${PROMPT_DIR:-prompts/macsql/default}"
FLUSH_EVERY="${FLUSH_EVERY:-20}"
LIMIT="${LIMIT:-}"
MAX_REFINE_ROUNDS="${MAX_REFINE_ROUNDS:-3}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-60}"
VALUE_EXAMPLES="${VALUE_EXAMPLES:-5}"
AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-auto}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-120}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-8}"
RUN_EVAL="${RUN_EVAL:-0}"

IFS=',' read -r -a ALL_BENCHMARKS <<< "${BENCHMARKS}"
IFS=',' read -r -a ALL_RUNS <<< "${RUNS}"
IFS=',' read -r -a ALL_SEEDS <<< "${SEEDS}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

benchmark_eval_name() {
  case "$1:$2" in
    spider_data:dev) echo "spider_dev" ;;
    spider_data:test) echo "spider_test" ;;
    spider_syn:test) echo "spider_syn_test" ;;
    spider_realistic:test) echo "spider_realistic_test" ;;
    spider_dk:test) echo "spider_dk_test" ;;
    *) echo "" ;;
  esac
}

run_one() {
  local run_name="$1"
  local benchmark="$2"
  local seed="$3"
  local output_dir output_path
  local selector_model decomposer_model refiner_model student_vllm_model student_vllm_base_url
  local -a cmd

  if [[ "${run_name}" == "teacher_lora" ]]; then
    selector_model="teacher"
    decomposer_model="teacher"
    refiner_model="teacher"
    student_vllm_model="${STUDENT_BASE_VLLM_MODEL}"
    student_vllm_base_url="${STUDENT_SFT_VLLM_BASE_URL}"
  else
    selector_model="student"
    decomposer_model="student"
    refiner_model="student"
    case "${run_name}" in
      student_sft)
        student_vllm_base_url="${STUDENT_SFT_VLLM_BASE_URL}"
        student_vllm_model="${STUDENT_SFT_VLLM_MODEL}"
        ;;
      csd)
        student_vllm_base_url="${CSD_VLLM_BASE_URL}"
        student_vllm_model="${CSD_VLLM_MODEL}"
        ;;
      distillm)
        student_vllm_base_url="${DISTILLM_VLLM_BASE_URL}"
        student_vllm_model="${DISTILLM_VLLM_MODEL}"
        ;;
      synid_sql)
        student_vllm_base_url="${SYNID_SQL_VLLM_BASE_URL}"
        student_vllm_model="${SYNID_SQL_VLLM_MODEL}"
        ;;
      *)
        student_vllm_base_url="${STUDENT_VLLM_BASE_URL}"
        student_vllm_model="${run_name}"
        ;;
    esac
  fi

  output_dir="${OUTPUT_ROOT}/${run_name}/${benchmark}/seed${seed}"
  output_path="${output_dir}/${benchmark}_${SPLIT}_sql_result.json"
  mkdir -p "${output_dir}"

  cmd=(
    "${PYTHON_BIN}" scripts/macsql/run_macsql.py
    --benchmark "${benchmark}"
    --split "${SPLIT}"
    --db "${DB}"
    --prompt-dir "${PROMPT_DIR}"
    --output_path "${output_path}"
    --flush-every "${FLUSH_EVERY}"
    --model-backend vllm
    --teacher-vllm-base-url "${TEACHER_VLLM_BASE_URL}"
    --student-vllm-base-url "${student_vllm_base_url}"
    --teacher-vllm-model "${TEACHER_VLLM_MODEL}"
    --student-vllm-model "${student_vllm_model}"
    --vllm-api-key "${VLLM_API_KEY}"
    --vllm-timeout "${VLLM_TIMEOUT}"
    --vllm-concurrency "${VLLM_CONCURRENCY}"
    --selector-model "${selector_model}"
    --decomposer-model "${decomposer_model}"
    --refiner-model "${refiner_model}"
    --max-refine-rounds "${MAX_REFINE_ROUNDS}"
    --execution-timeout "${EXECUTION_TIMEOUT}"
    --value-examples "${VALUE_EXAMPLES}"
    --agent-batch-size "${AGENT_BATCH_SIZE}"
    --seed "${seed}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
  )
  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  echo "[macsql-vllm] run=${run_name} benchmark=${benchmark} seed=${seed} out=${output_path}"
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

format_and_eval() {
  local run_name="$1"
  local benchmark="$2"
  local seed="$3"
  local output_dir formatted_dir prefix eval_name
  output_dir="${OUTPUT_ROOT}/${run_name}/${benchmark}/seed${seed}"
  formatted_dir="${output_dir}/formatted_data"
  prefix="${benchmark}_${SPLIT}"
  eval_name="$(benchmark_eval_name "${benchmark}" "${SPLIT}")"

  "${PYTHON_BIN}" scripts/format_spider_infer_results.py \
    --input-dir "${output_dir}" \
    --input-glob "${prefix}_sql_result.json" \
    --output-dir "${formatted_dir}"

  if [[ "${RUN_EVAL}" == "1" && -n "${eval_name}" ]]; then
    "${PYTHON_BIN}" src/evaluator/run_benchmark.py \
      --benchmark "${eval_name}" \
      --pred "${formatted_dir}/${prefix}.pred.sql" \
      --gold "${formatted_dir}/${prefix}.gold.sql" \
      --etype exec \
      --exec_timeout "${EXECUTION_TIMEOUT}"
  fi
}

for raw_run in "${ALL_RUNS[@]}"; do
  run_name="$(trim "${raw_run}")"
  [[ -z "${run_name}" ]] && continue
  for raw_benchmark in "${ALL_BENCHMARKS[@]}"; do
    benchmark="$(trim "${raw_benchmark}")"
    [[ -z "${benchmark}" ]] && continue
    for raw_seed in "${ALL_SEEDS[@]}"; do
      seed="$(trim "${raw_seed}")"
      [[ -z "${seed}" ]] && continue
      run_one "${run_name}" "${benchmark}" "${seed}"
      format_and_eval "${run_name}" "${benchmark}" "${seed}"
    done
  done
done
