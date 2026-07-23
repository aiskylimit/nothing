#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TEACHER_BASE="${TEACHER_BASE:-Qwen/Qwen3-4B-Instruct-2507}"
STUDENT_BASE="${STUDENT_BASE:-Qwen/Qwen3-0.6B}"

TEACHER_LORA="${TEACHER_LORA:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090}"
STUDENT_SFT="${STUDENT_SFT:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/sft_sft_qwen3_0.6b_spider/e5-bs4-lr5e-05-G4-N2-NN1/1090}"
DISTILLM_LORA="${DISTILLM_LORA:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/distillm_train_0.6b_4b_spider_adaptive-srkl_e5-bs2-lr0.0001-G8-N2-NN1-kd0.7-lora-16-64-0.1/872}"
CSD_LORA="${CSD_LORA:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/csd_train_0.6b_4b_kd0.6_spider_e5-bs2-lr0.0001-G8-N2-NN1-kd0.6-lora-16-64-0.1/654}"
SYNID_SQL_LORA="${SYNID_SQL_LORA:-https://huggingface.co/Dream-AI-HUST/synid_ckpt/tree/main/results/qwen3/synid_ce_keywords_weight_lora_218_train_g01_spider_synid_datalora218-e5-bs4-lr0.0001-G1-gridG01-k1-kd0.7-csd-tau0.05-a0.3-b0.3-k1_last_s27_t35-poolsc-keywords-lambda2.0-lora-16-64-0.1/4375}"

BENCHMARKS="${BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/macsql_agent}"
PROMPT_DIR="${PROMPT_DIR:-prompts/macsql/default}"
SELECTOR_MODEL="${SELECTOR_MODEL:-student}"
DECOMPOSER_MODEL="${DECOMPOSER_MODEL:-teacher}"
REFINER_MODEL="${REFINER_MODEL:-student}"
MAX_REFINE_ROUNDS="${MAX_REFINE_ROUNDS:-3}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-30}"
VALUE_EXAMPLES="${VALUE_EXAMPLES:-5}"
FLUSH_EVERY="${FLUSH_EVERY:-20}"
LIMIT="${LIMIT:-}"
SEEDS="${SEEDS:-10,42,50,100,1234}"
RUN_EVAL="${RUN_EVAL:-1}"
UV_SYNC="${UV_SYNC:-0}"

if [[ "${UV_SYNC}" == "1" ]]; then
  uv sync
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

run_cmd() {
  echo "+ $*"
  "$@"
}

benchmark_eval_name() {
  case "$1:$2" in
    spider_data:dev) echo "spider_dev" ;;
    spider_data:test) echo "spider_test" ;;
    spider_syn:test) echo "spider_syn_test" ;;
    spider_realistic:test) echo "spider_realistic_test" ;;
    spider_dk:test) echo "spider_dk_test" ;;
    *)
      echo ""
      ;;
  esac
}

run_agent() {
  local run_name="$1"
  local student_sft="$2"
  local student_loras="$3"
  local benchmark="$4"
  local seed="$5"
  local output_dir="${OUTPUT_ROOT}/${run_name}/${benchmark}/seed${seed}"
  local output_path="${output_dir}/${benchmark}_${SPLIT}_sql_result.json"
  local args=(
    scripts/macsql/run_macsql.py
    --benchmark "${benchmark}"
    --split "${SPLIT}"
    --prompt-dir "${PROMPT_DIR}"
    --output_path "${output_path}"
    --flush-every "${FLUSH_EVERY}"
    --teacher-base "${TEACHER_BASE}"
    --teacher-lora-adapter "${TEACHER_LORA}"
    --student-base "${STUDENT_BASE}"
    --student-sft-ckpt "${student_sft}"
    --device "${DEVICE}"
    --selector-model "${SELECTOR_MODEL}"
    --decomposer-model "${DECOMPOSER_MODEL}"
    --refiner-model "${REFINER_MODEL}"
    --max-refine-rounds "${MAX_REFINE_ROUNDS}"
    --execution-timeout "${EXECUTION_TIMEOUT}"
    --value-examples "${VALUE_EXAMPLES}"
    --seed "${seed}"
  )

  if [[ -n "${student_loras}" ]]; then
    args+=(--student-lora-adapters "${student_loras}")
  fi
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi

  mkdir -p "${output_dir}"
  run_cmd python "${args[@]}"
}

format_and_eval() {
  local run_name="$1"
  local benchmark="$2"
  local seed="$3"
  local output_dir="${OUTPUT_ROOT}/${run_name}/${benchmark}/seed${seed}"
  local formatted_dir="${output_dir}/formatted_data"
  local prefix="${benchmark}_${SPLIT}"
  local eval_name
  eval_name="$(benchmark_eval_name "${benchmark}" "${SPLIT}")"

  run_cmd python scripts/format_spider_infer_results.py \
    --input-dir "${output_dir}" \
    --input-glob "${prefix}_sql_result.json" \
    --output-dir "${formatted_dir}"

  if [[ "${RUN_EVAL}" != "1" || -z "${eval_name}" ]]; then
    return
  fi

  run_cmd python src/evaluator/run_benchmark.py \
    --benchmark "${eval_name}" \
    --pred "${formatted_dir}/${prefix}.pred.sql" \
    --gold "${formatted_dir}/${prefix}.gold.sql" \
    --etype exec \
    --exec_timeout "${EXECUTION_TIMEOUT}"
}

run_suite() {
  local run_name="$1"
  local student_sft="$2"
  local student_loras="$3"
  IFS=',' read -ra benchmark_list <<< "${BENCHMARKS}"
  IFS=',' read -ra seed_list <<< "${SEEDS}"

  for benchmark in "${benchmark_list[@]}"; do
    for seed in "${seed_list[@]}"; do
      seed="${seed//[[:space:]]/}"
      if [[ -z "${seed}" ]]; then
        continue
      fi
      run_agent "${run_name}" "${student_sft}" "${student_loras}" "${benchmark}" "${seed}"
      format_and_eval "${run_name}" "${benchmark}" "${seed}"
    done
  done
}

run_suite "student_sft" "${STUDENT_SFT}" ""
run_suite "distillm" "${STUDENT_SFT}" "${DISTILLM_LORA}"
run_suite "csd" "${STUDENT_SFT}" "${CSD_LORA}"
run_suite "synid_sql" "${STUDENT_SFT}" "${SYNID_SQL_LORA}"
