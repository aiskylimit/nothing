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
RUNS="${RUNS:-teacher_lora,student_sft,distillm,csd,synid_sql}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/macsql_agent}"
PROMPT_DIR="${PROMPT_DIR:-prompts/macsql/default}"
TEACHER_SELECTOR_MODEL="${TEACHER_SELECTOR_MODEL:-teacher}"
TEACHER_DECOMPOSER_MODEL="${TEACHER_DECOMPOSER_MODEL:-teacher}"
TEACHER_REFINER_MODEL="${TEACHER_REFINER_MODEL:-teacher}"
STUDENT_SELECTOR_MODEL="${STUDENT_SELECTOR_MODEL:-student}"
STUDENT_DECOMPOSER_MODEL="${STUDENT_DECOMPOSER_MODEL:-student}"
STUDENT_REFINER_MODEL="${STUDENT_REFINER_MODEL:-student}"
MAX_REFINE_ROUNDS="${MAX_REFINE_ROUNDS:-3}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-30}"
VALUE_EXAMPLES="${VALUE_EXAMPLES:-5}"
AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE:-1}"
FLUSH_EVERY="${FLUSH_EVERY:-20}"
LIMIT="${LIMIT:-}"
SEEDS="${SEEDS:-10,42,50,100,1234}"
RUN_EVAL="${RUN_EVAL:-1}"
UV_SYNC="${UV_SYNC:-1}"
SETUP_BENCHMARKS="${SETUP_BENCHMARKS:-1}"
UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"
HF_UPLOAD_REPO_ID="${HF_UPLOAD_REPO_ID:-distillation-sql/nothing_agent}"
HF_UPLOAD_REPO_TYPE="${HF_UPLOAD_REPO_TYPE:-model}"
HF_UPLOAD_PATH_PREFIX="${HF_UPLOAD_PATH_PREFIX:-${OUTPUT_ROOT}}"

if [[ "${UV_SYNC}" == "1" ]]; then
  uv sync
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

setup_hf_auth() {
  if [[ "${UPLOAD_TO_HF}" != "1" ]]; then
    return
  fi
  if [[ -n "${HF_UPLOAD_TOKEN:-}" ]]; then
    hf auth login --token "${HF_UPLOAD_TOKEN}"
  fi
}

setup_benchmarks() {
  if [[ "${SETUP_BENCHMARKS}" != "1" ]]; then
    return
  fi
  if [[ -d "benchmarks/spider_data" && -d "benchmarks/spider_syn" && -d "benchmarks/spider_realistic" && -d "benchmarks/spider_dk" ]]; then
    return
  fi

  # Download benchmark assets if the local benchmark folders are missing.
  hf download Dream-AI-HUST/sql_benchmarks \
    --repo-type dataset \
    --local-dir .
  unzip -o benchmarks.zip
  unzip -o data.zip
}

setup_hf_auth
setup_benchmarks

python -c "import nltk; nltk.download('punkt_tab')"

IFS=',' read -ra ALL_BENCHMARKS <<< "${BENCHMARKS}"
IFS=',' read -ra ALL_SEEDS <<< "${SEEDS}"
IFS=',' read -ra ALL_RUNS <<< "${RUNS}"
TOTAL_JOBS=$((${#ALL_RUNS[@]} * ${#ALL_BENCHMARKS[@]} * ${#ALL_SEEDS[@]}))
JOB_INDEX=0

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

should_run_suite() {
  local wanted="$1"
  local item
  for item in "${ALL_RUNS[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ "${item}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

run_agent() {
  local run_name="$1"
  local student_sft="$2"
  local student_loras="$3"
  local benchmark="$4"
  local seed="$5"
  local selector_model="$6"
  local decomposer_model="$7"
  local refiner_model="$8"
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
    --selector-model "${selector_model}"
    --decomposer-model "${decomposer_model}"
    --refiner-model "${refiner_model}"
    --max-refine-rounds "${MAX_REFINE_ROUNDS}"
    --execution-timeout "${EXECUTION_TIMEOUT}"
    --value-examples "${VALUE_EXAMPLES}"
    --agent-batch-size "${AGENT_BATCH_SIZE}"
    --seed "${seed}"
  )

  if [[ -n "${student_loras}" ]]; then
    args+=(--student-lora-adapters "${student_loras}")
  fi
  if [[ -n "${LIMIT}" ]]; then
    args+=(--limit "${LIMIT}")
  fi

  mkdir -p "${output_dir}"
  JOB_INDEX=$((JOB_INDEX + 1))
  echo "[agent-job ${JOB_INDEX}/${TOTAL_JOBS}] run=${run_name} benchmark=${benchmark} split=${SPLIT} seed=${seed} selector=${selector_model} decomposer=${decomposer_model} refiner=${refiner_model} batch=${AGENT_BATCH_SIZE}"
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

upload_job_output() {
  local run_name="$1"
  local benchmark="$2"
  local seed="$3"
  local output_dir="${OUTPUT_ROOT}/${run_name}/${benchmark}/seed${seed}"
  local repo_path="${HF_UPLOAD_PATH_PREFIX}/${run_name}/${benchmark}/seed${seed}"

  if [[ "${UPLOAD_TO_HF}" != "1" ]]; then
    return
  fi
  if [[ ! -d "${output_dir}" ]]; then
    echo "[upload-skip] missing output_dir=${output_dir}"
    return
  fi

  echo "[upload-start] local=${output_dir} repo=${HF_UPLOAD_REPO_ID}:${repo_path}"
  run_cmd hf upload "${HF_UPLOAD_REPO_ID}" "${output_dir}" "${repo_path}" \
    --repo-type "${HF_UPLOAD_REPO_TYPE}"
  echo "[upload-done] local=${output_dir}"
}

run_suite() {
  local run_name="$1"
  local student_sft="$2"
  local student_loras="$3"
  local selector_model="$4"
  local decomposer_model="$5"
  local refiner_model="$6"
  IFS=',' read -ra benchmark_list <<< "${BENCHMARKS}"
  IFS=',' read -ra seed_list <<< "${SEEDS}"

  for benchmark in "${benchmark_list[@]}"; do
    for seed in "${seed_list[@]}"; do
      seed="${seed//[[:space:]]/}"
      if [[ -z "${seed}" ]]; then
        continue
      fi
      run_agent "${run_name}" "${student_sft}" "${student_loras}" "${benchmark}" "${seed}" "${selector_model}" "${decomposer_model}" "${refiner_model}"
      format_and_eval "${run_name}" "${benchmark}" "${seed}"
      upload_job_output "${run_name}" "${benchmark}" "${seed}"
    done
  done
}

if should_run_suite "teacher_lora"; then
  run_suite "teacher_lora" "" "" "${TEACHER_SELECTOR_MODEL}" "${TEACHER_DECOMPOSER_MODEL}" "${TEACHER_REFINER_MODEL}"
fi
if should_run_suite "student_sft"; then
  run_suite "student_sft" "${STUDENT_SFT}" "" "${STUDENT_SELECTOR_MODEL}" "${STUDENT_DECOMPOSER_MODEL}" "${STUDENT_REFINER_MODEL}"
fi
if should_run_suite "distillm"; then
  run_suite "distillm" "${STUDENT_SFT}" "${DISTILLM_LORA}" "${STUDENT_SELECTOR_MODEL}" "${STUDENT_DECOMPOSER_MODEL}" "${STUDENT_REFINER_MODEL}"
fi
if should_run_suite "csd"; then
  run_suite "csd" "${STUDENT_SFT}" "${CSD_LORA}" "${STUDENT_SELECTOR_MODEL}" "${STUDENT_DECOMPOSER_MODEL}" "${STUDENT_REFINER_MODEL}"
fi
if should_run_suite "synid_sql"; then
  run_suite "synid_sql" "${STUDENT_SFT}" "${SYNID_SQL_LORA}" "${STUDENT_SELECTOR_MODEL}" "${STUDENT_DECOMPOSER_MODEL}" "${STUDENT_REFINER_MODEL}"
fi
