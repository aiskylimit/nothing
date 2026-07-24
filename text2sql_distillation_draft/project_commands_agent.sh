#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

# vLLM servers live on GPUs 0, 1, 2, 3.
TEACHER_GPUS="${TEACHER_GPUS:-0}"
STUDENT_SFT_GPUS="${STUDENT_SFT_GPUS:-1}"
STUDENT_CSD_DISTILLM_GPUS="${STUDENT_CSD_DISTILLM_GPUS:-2}"
STUDENT_SYNID_SQL_GPUS="${STUDENT_SYNID_SQL_GPUS:-3}"
TEACHER_PORT="${TEACHER_PORT:-8101}"
STUDENT_SFT_PORT="${STUDENT_SFT_PORT:-8102}"
STUDENT_CSD_DISTILLM_PORT="${STUDENT_CSD_DISTILLM_PORT:-8103}"
STUDENT_SYNID_SQL_PORT="${STUDENT_SYNID_SQL_PORT:-8104}"

# MAC-SQL client, formatting, and SQLite evaluation do not need GPU.
INFER_EVAL_GPUS="${INFER_EVAL_GPUS:-}"

TEACHER_BASE="${TEACHER_BASE:-Qwen/Qwen3-4B-Instruct-2507}"
STUDENT_BASE="${STUDENT_BASE:-Qwen/Qwen3-0.6B}"
TEACHER_LORA="${TEACHER_LORA:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090}"
STUDENT_SFT="${STUDENT_SFT:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/sft_sft_qwen3_0.6b_spider/e5-bs4-lr5e-05-G4-N2-NN1/1090}"
DISTILLM_LORA="${DISTILLM_LORA:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/distillm_train_0.6b_4b_spider_adaptive-srkl_e5-bs2-lr0.0001-G8-N2-NN1-kd0.7-lora-16-64-0.1/872}"
CSD_LORA="${CSD_LORA:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/csd_train_0.6b_4b_kd0.6_spider_e5-bs2-lr0.0001-G8-N2-NN1-kd0.6-lora-16-64-0.1/654}"
SYNID_SQL_LORA="${SYNID_SQL_LORA:-https://huggingface.co/Dream-AI-HUST/synid_ckpt/tree/main/results/qwen3/synid_ce_keywords_weight_lora_218_train_g01_spider_synid_datalora218-e5-bs4-lr0.0001-G1-gridG01-k1-kd0.7-csd-tau0.05-a0.3-b0.3-k1_last_s27_t35-poolsc-keywords-lambda2.0-lora-16-64-0.1/4375}"

TEACHER_VLLM_BASE_URL="${TEACHER_VLLM_BASE_URL:-http://localhost:${TEACHER_PORT}/v1}"
STUDENT_SFT_VLLM_BASE_URL="${STUDENT_SFT_VLLM_BASE_URL:-http://localhost:${STUDENT_SFT_PORT}/v1}"
CSD_VLLM_BASE_URL="${CSD_VLLM_BASE_URL:-http://localhost:${STUDENT_CSD_DISTILLM_PORT}/v1}"
DISTILLM_VLLM_BASE_URL="${DISTILLM_VLLM_BASE_URL:-http://localhost:${STUDENT_CSD_DISTILLM_PORT}/v1}"
SYNID_SQL_VLLM_BASE_URL="${SYNID_SQL_VLLM_BASE_URL:-http://localhost:${STUDENT_SYNID_SQL_PORT}/v1}"
TEACHER_VLLM_MODEL="${TEACHER_VLLM_MODEL:-teacher_lora}"
STUDENT_SFT_VLLM_MODEL="${STUDENT_SFT_VLLM_MODEL:-student_sft}"
CSD_VLLM_MODEL="${CSD_VLLM_MODEL:-csd}"
DISTILLM_VLLM_MODEL="${DISTILLM_VLLM_MODEL:-distillm}"
SYNID_SQL_VLLM_MODEL="${SYNID_SQL_VLLM_MODEL:-synid_sql}"

BENCHMARKS="${BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
RUNS="${RUNS:-teacher_lora,student_sft,distillm,csd,synid_sql}"
SEEDS="${SEEDS:-10,42,50,100,1234}"
SPLIT="${SPLIT:-test}"
DB="${DB:-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/macsql_vllm}"
PROMPT_DIR="${PROMPT_DIR:-prompts/macsql/default}"
MAX_REFINE_ROUNDS="${MAX_REFINE_ROUNDS:-3}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-30}"
VALUE_EXAMPLES="${VALUE_EXAMPLES:-5}"
AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE:-8}"
FLUSH_EVERY="${FLUSH_EVERY:-20}"
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-auto}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-120}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-8}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

UV_SYNC="${UV_SYNC:-1}"
SETUP_BENCHMARKS="${SETUP_BENCHMARKS:-1}"
DOWNLOAD_NLTK="${DOWNLOAD_NLTK:-1}"
START_SERVERS="${START_SERVERS:-1}"
STOP_SERVERS_ON_EXIT="${STOP_SERVERS_ON_EXIT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SERVE_STRATEGY="${SERVE_STRATEGY:-all_at_once_4gpu}"
LORA_BASE_MODEL="${LORA_BASE_MODEL:-${STUDENT_BASE}}"

run_cmd() {
  echo "+ $*"
  "$@"
}

setup_env() {
  if [[ "${UV_SYNC}" == "1" ]]; then
    run_cmd uv sync
  fi

  if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi

  if [[ "${SETUP_BENCHMARKS}" == "1" ]]; then
    if [[ ! -d "benchmarks/spider_data" || ! -d "benchmarks/spider_syn" || ! -d "benchmarks/spider_realistic" || ! -d "benchmarks/spider_dk" ]]; then
      run_cmd hf download Dream-AI-HUST/sql_benchmarks --repo-type dataset --local-dir .
      run_cmd unzip -o benchmarks.zip
      run_cmd unzip -o data.zip
    fi
  fi

  if [[ "${DOWNLOAD_NLTK}" == "1" ]]; then
    run_cmd "${PYTHON_BIN}" -c "import nltk; nltk.download('punkt_tab')"
  fi
}

start_all_vllm_servers() {
  if [[ "${START_SERVERS}" != "1" ]]; then
    return
  fi

  echo "[serve-4gpu] gpu0=teacher_lora gpu1=student_sft gpu2=csd,distillm gpu3=synid_sql"
  SERVER_NAME="teacher_lora" \
  SERVER_GPUS="${TEACHER_GPUS}" \
  SERVER_PORT="${TEACHER_PORT}" \
  SERVER_MODEL="${TEACHER_BASE}" \
  SERVER_SERVED_MODEL_NAME="teacher_base" \
  SERVER_LORAS="teacher_lora=${TEACHER_LORA}" \
  VLLM_API_KEY="${VLLM_API_KEY}" \
  bash scripts/macsql/serve_vllm_models.sh start-one

  SERVER_NAME="student_sft" \
  SERVER_GPUS="${STUDENT_SFT_GPUS}" \
  SERVER_PORT="${STUDENT_SFT_PORT}" \
  SERVER_MODEL="${STUDENT_SFT}" \
  SERVER_SERVED_MODEL_NAME="student_sft" \
  SERVER_LORAS="" \
  VLLM_API_KEY="${VLLM_API_KEY}" \
  bash scripts/macsql/serve_vllm_models.sh start-one

  SERVER_NAME="student_csd_distillm" \
  SERVER_GPUS="${STUDENT_CSD_DISTILLM_GPUS}" \
  SERVER_PORT="${STUDENT_CSD_DISTILLM_PORT}" \
  SERVER_MODEL="${LORA_BASE_MODEL}" \
  SERVER_SERVED_MODEL_NAME="student_base_csd_distillm" \
  SERVER_LORAS="csd=${CSD_LORA},distillm=${DISTILLM_LORA}" \
  VLLM_API_KEY="${VLLM_API_KEY}" \
  bash scripts/macsql/serve_vllm_models.sh start-one

  SERVER_NAME="student_synid_sql" \
  SERVER_GPUS="${STUDENT_SYNID_SQL_GPUS}" \
  SERVER_PORT="${STUDENT_SYNID_SQL_PORT}" \
  SERVER_MODEL="${LORA_BASE_MODEL}" \
  SERVER_SERVED_MODEL_NAME="student_base_synid_sql" \
  SERVER_LORAS="synid_sql=${SYNID_SQL_LORA}" \
  VLLM_API_KEY="${VLLM_API_KEY}" \
  bash scripts/macsql/serve_vllm_models.sh start-one
}

stop_vllm_servers() {
  local force="${1:-0}"
  if [[ "${START_SERVERS}" == "1" && ( "${STOP_SERVERS_ON_EXIT}" == "1" || "${force}" == "1" ) ]]; then
    SERVER_NAME="teacher_lora" bash scripts/macsql/serve_vllm_models.sh stop-one || true
    SERVER_NAME="student_sft" bash scripts/macsql/serve_vllm_models.sh stop-one || true
    SERVER_NAME="student_csd_distillm" bash scripts/macsql/serve_vllm_models.sh stop-one || true
    SERVER_NAME="student_synid_sql" bash scripts/macsql/serve_vllm_models.sh stop-one || true
    bash scripts/macsql/serve_vllm_models.sh stop || true
  fi
}

run_infer_eval_suite() {
  local run_list="$1"
  echo "[infer-eval] runs=${run_list} CUDA_VISIBLE_DEVICES=${INFER_EVAL_GPUS}"
  CUDA_VISIBLE_DEVICES="${INFER_EVAL_GPUS}" \
  TEACHER_VLLM_BASE_URL="${TEACHER_VLLM_BASE_URL}" \
  STUDENT_SFT_VLLM_BASE_URL="${STUDENT_SFT_VLLM_BASE_URL}" \
  CSD_VLLM_BASE_URL="${CSD_VLLM_BASE_URL}" \
  DISTILLM_VLLM_BASE_URL="${DISTILLM_VLLM_BASE_URL}" \
  SYNID_SQL_VLLM_BASE_URL="${SYNID_SQL_VLLM_BASE_URL}" \
  TEACHER_VLLM_MODEL="${TEACHER_VLLM_MODEL}" \
  STUDENT_SFT_VLLM_MODEL="${STUDENT_SFT_VLLM_MODEL}" \
  CSD_VLLM_MODEL="${CSD_VLLM_MODEL}" \
  DISTILLM_VLLM_MODEL="${DISTILLM_VLLM_MODEL}" \
  SYNID_SQL_VLLM_MODEL="${SYNID_SQL_VLLM_MODEL}" \
  BENCHMARKS="${BENCHMARKS}" \
  RUNS="${run_list}" \
  SEEDS="${SEEDS}" \
  SPLIT="${SPLIT}" \
  DB="${DB}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  PROMPT_DIR="${PROMPT_DIR}" \
  MAX_REFINE_ROUNDS="${MAX_REFINE_ROUNDS}" \
  EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT}" \
  VALUE_EXAMPLES="${VALUE_EXAMPLES}" \
  AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE}" \
  FLUSH_EVERY="${FLUSH_EVERY}" \
  LIMIT="${LIMIT}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_P="${TOP_P}" \
  TOP_K="${TOP_K}" \
  VLLM_API_KEY="${VLLM_API_KEY}" \
  VLLM_TIMEOUT="${VLLM_TIMEOUT}" \
  VLLM_CONCURRENCY="${VLLM_CONCURRENCY}" \
  RUN_EVAL="${RUN_EVAL}" \
  bash scripts/macsql/run_macsql_vllm_suite.sh
}

run_has() {
  local wanted="$1"
  local raw item
  IFS=',' read -r -a raw_runs <<< "${RUNS}"
  for raw in "${raw_runs[@]}"; do
    item="${raw//[[:space:]]/}"
    if [[ "${item}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

lora_run_subset() {
  local names=()
  run_has "distillm" && names+=("distillm")
  run_has "csd" && names+=("csd")
  run_has "synid_sql" && names+=("synid_sql")
  local IFS=','
  echo "${names[*]}"
}

start_teacher_phase() {
  if [[ "${START_SERVERS}" != "1" ]]; then
    return
  fi
  echo "[phase-serve] teacher_lora on gpu=${TEACHER_GPUS}"
  RUN_TEACHER=1 \
  RUN_STUDENT=0 \
  TEACHER_GPUS="${TEACHER_GPUS}" \
  TEACHER_PORT="${TEACHER_PORT}" \
  TEACHER_BASE="${TEACHER_BASE}" \
  TEACHER_LORAS="teacher_lora=${TEACHER_LORA}" \
  VLLM_API_KEY="${VLLM_API_KEY}" \
  bash scripts/macsql/serve_vllm_models.sh start
}

start_student_sft_phase() {
  if [[ "${START_SERVERS}" != "1" ]]; then
    return
  fi
  echo "[phase-serve] student_sft on gpu=${STUDENT_SFT_GPUS}"
  RUN_TEACHER=0 \
  RUN_STUDENT=1 \
  STUDENT_GPUS="${STUDENT_SFT_GPUS}" \
  STUDENT_PORT="${STUDENT_SFT_PORT}" \
  STUDENT_BASE_MODEL="${STUDENT_SFT}" \
  STUDENT_SERVED_MODEL_NAME="student_sft" \
  STUDENT_LORAS="" \
  VLLM_API_KEY="${VLLM_API_KEY}" \
  bash scripts/macsql/serve_vllm_models.sh start
}

start_lora_phase() {
  if [[ "${START_SERVERS}" != "1" ]]; then
    return
  fi
  echo "[phase-serve] student base + LoRA adapters on gpu=${STUDENT_CSD_DISTILLM_GPUS} base=${LORA_BASE_MODEL}"
  RUN_TEACHER=0 \
  RUN_STUDENT=1 \
  STUDENT_GPUS="${STUDENT_CSD_DISTILLM_GPUS}" \
  STUDENT_PORT="${STUDENT_CSD_DISTILLM_PORT}" \
  STUDENT_BASE_MODEL="${LORA_BASE_MODEL}" \
  STUDENT_SERVED_MODEL_NAME="student_base" \
  STUDENT_LORAS="distillm=${DISTILLM_LORA},csd=${CSD_LORA},synid_sql=${SYNID_SQL_LORA}" \
  VLLM_API_KEY="${VLLM_API_KEY}" \
  bash scripts/macsql/serve_vllm_models.sh start
}

run_phased_suite() {
  local lora_runs
  lora_runs="$(lora_run_subset)"

  if run_has "teacher_lora"; then
    start_teacher_phase
    run_infer_eval_suite "teacher_lora"
    stop_vllm_servers 1
  fi

  if run_has "student_sft"; then
    start_student_sft_phase
    run_infer_eval_suite "student_sft"
    stop_vllm_servers 1
  fi

  if [[ -n "${lora_runs}" ]]; then
    start_lora_phase
    CSD_VLLM_BASE_URL="http://localhost:${STUDENT_CSD_DISTILLM_PORT}/v1" \
    DISTILLM_VLLM_BASE_URL="http://localhost:${STUDENT_CSD_DISTILLM_PORT}/v1" \
    SYNID_SQL_VLLM_BASE_URL="http://localhost:${STUDENT_CSD_DISTILLM_PORT}/v1" \
    run_infer_eval_suite "${lora_runs}"
    stop_vllm_servers 1
  fi
}

trap stop_vllm_servers EXIT

setup_env
case "${SERVE_STRATEGY}" in
  all_at_once_4gpu)
    start_all_vllm_servers
    run_infer_eval_suite "${RUNS}"
    ;;
  phased)
    run_phased_suite
    ;;
  all_at_once)
    start_all_vllm_servers
    run_infer_eval_suite "${RUNS}"
    ;;
  *)
    echo "Unknown SERVE_STRATEGY=${SERVE_STRATEGY}. Use all_at_once_4gpu, phased, or all_at_once." >&2
    exit 1
    ;;
esac
