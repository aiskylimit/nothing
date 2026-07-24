#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

COMMAND="${1:-start}"
PYTHON_BIN="${PYTHON:-python}"

HOST="${HOST:-0.0.0.0}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-8101}"
STUDENT_PORT="${STUDENT_PORT:-8102}"
TEACHER_GPUS="${TEACHER_GPUS:-0}"
STUDENT_GPUS="${STUDENT_GPUS:-1}"
RUN_TEACHER="${RUN_TEACHER:-1}"
RUN_STUDENT="${RUN_STUDENT:-1}"

PID_DIR="${PID_DIR:-run_logs/vllm/pids}"
LOG_DIR="${LOG_DIR:-run_logs/vllm/logs}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-900}"
RESOLVE_HF="${RESOLVE_HF:-1}"

TEACHER_BASE="${TEACHER_BASE:-Qwen/Qwen3-4B-Instruct-2507}"
STUDENT_BASE="${STUDENT_BASE:-Qwen/Qwen3-0.6B}"

TEACHER_LORA="${TEACHER_LORA:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090}"
STUDENT_SFT="${STUDENT_SFT:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/sft_sft_qwen3_0.6b_spider/e5-bs4-lr5e-05-G4-N2-NN1/1090}"
DISTILLM_LORA="${DISTILLM_LORA:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/distillm_train_0.6b_4b_spider_adaptive-srkl_e5-bs2-lr0.0001-G8-N2-NN1-kd0.7-lora-16-64-0.1/872}"
CSD_LORA="${CSD_LORA:-https://huggingface.co/distillation-sql/baselines/tree/main/qwen3/csd_train_0.6b_4b_kd0.6_spider_e5-bs2-lr0.0001-G8-N2-NN1-kd0.6-lora-16-64-0.1/654}"
SYNID_SQL_LORA="${SYNID_SQL_LORA:-https://huggingface.co/Dream-AI-HUST/synid_ckpt/tree/main/results/qwen3/synid_ce_keywords_weight_lora_218_train_g01_spider_synid_datalora218-e5-bs4-lr0.0001-G1-gridG01-k1-kd0.7-csd-tau0.05-a0.3-b0.3-k1_last_s27_t35-poolsc-keywords-lambda2.0-lora-16-64-0.1/4375}"

TEACHER_SERVED_MODEL_NAME="${TEACHER_SERVED_MODEL_NAME:-teacher_base}"
STUDENT_SERVED_MODEL_NAME="${STUDENT_SERVED_MODEL_NAME:-student_sft}"
STUDENT_BASE_MODEL="${STUDENT_BASE_MODEL:-${STUDENT_SFT}}"

TEACHER_LORAS="${TEACHER_LORAS:-teacher_lora=${TEACHER_LORA}}"
STUDENT_LORAS="${STUDENT_LORAS:-distillm=${DISTILLM_LORA},csd=${CSD_LORA},synid_sql=${SYNID_SQL_LORA}}"
EXTRA_TEACHER_LORAS="${EXTRA_TEACHER_LORAS:-}"
EXTRA_STUDENT_LORAS="${EXTRA_STUDENT_LORAS:-}"

DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MAX_LORAS="${MAX_LORAS:-8}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-32}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

mkdir -p "${PID_DIR}" "${LOG_DIR}"

usage() {
  cat <<'EOF'
Usage: scripts/macsql/serve_vllm_models.sh [start|stop|status]
       SERVER_NAME=name SERVER_GPUS=0 SERVER_PORT=8101 SERVER_MODEL=model \
       SERVER_SERVED_MODEL_NAME=base_name SERVER_LORAS='lora_name=/path' \
       scripts/macsql/serve_vllm_models.sh start-one

Starts two OpenAI-compatible vLLM servers by default:
  teacher : Qwen3-4B base + teacher_lora on port 8101
  student : student_sft full checkpoint + distillm/csd/synid_sql LoRA adapters on port 8102

Important env overrides:
  TEACHER_GPUS=0 STUDENT_GPUS=1
  TEACHER_PORT=8101 STUDENT_PORT=8102
  TEACHER_LORAS='teacher_lora=/path/to/lora'
  STUDENT_BASE_MODEL=/path/to/full_sft_or_hf_repo
  STUDENT_LORAS='distillm=/path/a,csd=/path/b,synid_sql=/path/c'
  EXTRA_STUDENT_LORAS='ckpt123=/path/123,ckpt456=/path/456'
  TENSOR_PARALLEL_SIZE=1 MAX_MODEL_LEN=8192 MAX_LORA_RANK=64
EOF
}

pid_file() {
  local name="$1"
  printf '%s/%s.pid' "${PID_DIR}" "${name}"
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

resolve_source() {
  local source="$1"
  if [[ -z "${source}" ]]; then
    printf ''
    return
  fi
  if [[ "${RESOLVE_HF}" != "1" || -e "${source}" ]]; then
    printf '%s' "${source}"
    return
  fi
  "${PYTHON_BIN}" -c 'import sys; from infer import _resolve_ckpt_dir; print(_resolve_ckpt_dir(sys.argv[1], None)[0])' "${source}"
}

append_lora_modules() {
  local -n output_modules="$1"
  local specs="$2"
  local raw spec name source resolved
  IFS=',' read -r -a raw_specs <<< "${specs}"
  for raw in "${raw_specs[@]}"; do
    spec="$(trim "${raw}")"
    if [[ -z "${spec}" ]]; then
      continue
    fi
    if [[ "${spec}" != *=* ]]; then
      echo "Invalid LoRA spec '${spec}'. Expected name=path." >&2
      exit 1
    fi
    name="$(trim "${spec%%=*}")"
    source="$(trim "${spec#*=}")"
    if [[ -z "${name}" || -z "${source}" ]]; then
      echo "Invalid LoRA spec '${spec}'. Empty name or path." >&2
      exit 1
    fi
    resolved="$(resolve_source "${source}")"
    output_modules+=("${name}=${resolved}")
  done
}

wait_for_server() {
  local name="$1"
  local port="$2"
  local start_ts now elapsed
  start_ts="$(date +%s)"
  until curl -fsS "http://${HEALTH_HOST}:${port}/health" >/dev/null 2>&1; do
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if (( elapsed > WAIT_TIMEOUT_SECONDS )); then
      echo "[${name}] health check timed out after ${WAIT_TIMEOUT_SECONDS}s" >&2
      return 1
    fi
    echo "[${name}] waiting for http://${HEALTH_HOST}:${port}/health (${elapsed}s)"
    sleep 5
  done
  echo "[${name}] ready: http://${HEALTH_HOST}:${port}/v1"
  curl -fsS -H "Authorization: Bearer ${VLLM_API_KEY}" "http://${HEALTH_HOST}:${port}/v1/models" || true
  echo
}

start_server() {
  local name="$1"
  local gpus="$2"
  local port="$3"
  local model_source="$4"
  local served_model_name="$5"
  shift 5
  local lora_modules=("$@")
  local pid_path log_path resolved_model
  local -a cmd extra_args

  pid_path="$(pid_file "${name}")"
  log_path="${LOG_DIR}/${name}.log"
  if [[ -f "${pid_path}" ]] && kill -0 "$(cat "${pid_path}")" 2>/dev/null; then
    echo "[${name}] already running with pid $(cat "${pid_path}")"
    return
  fi

  resolved_model="$(resolve_source "${model_source}")"
  cmd=(
    vllm serve "${resolved_model}"
    --host "${HOST}"
    --port "${port}"
    --api-key "${VLLM_API_KEY}"
    --served-model-name "${served_model_name}"
    --dtype "${DTYPE}"
    --trust-remote-code
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
  )
  if [[ "${TENSOR_PARALLEL_SIZE}" != "1" ]]; then
    cmd+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
  fi
  if [[ "${DISABLE_THINKING}" == "1" ]]; then
    cmd+=(--default-chat-template-kwargs '{"enable_thinking": false}')
  fi
  if [[ "${#lora_modules[@]}" -gt 0 ]]; then
    cmd+=(
      --enable-lora
      --max-lora-rank "${MAX_LORA_RANK}"
      --max-loras "${MAX_LORAS}"
      --max-cpu-loras "${MAX_CPU_LORAS}"
      --lora-modules
    )
    cmd+=("${lora_modules[@]}")
  fi
  if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
    # Intentional word splitting for a raw vLLM escape hatch.
    read -r -a extra_args <<< "${EXTRA_VLLM_ARGS}"
    cmd+=("${extra_args[@]}")
  fi

  echo "[${name}] starting on GPUs=${gpus} port=${port}"
  echo "[${name}] log=${log_path}"
  echo "+ CUDA_VISIBLE_DEVICES=${gpus} ${cmd[*]}"
  CUDA_VISIBLE_DEVICES="${gpus}" nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  echo "$!" > "${pid_path}"
  wait_for_server "${name}" "${port}"
}

stop_server() {
  local name="$1"
  local pid_path
  pid_path="$(pid_file "${name}")"
  if [[ ! -f "${pid_path}" ]]; then
    echo "[${name}] no pid file"
    return
  fi
  local pid
  pid="$(cat "${pid_path}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[${name}] stopping pid=${pid}"
    kill "${pid}"
  else
    echo "[${name}] pid=${pid} is not running"
  fi
  rm -f "${pid_path}"
}

status_server() {
  local name="$1"
  local port="$2"
  local pid_path
  pid_path="$(pid_file "${name}")"
  if [[ -f "${pid_path}" ]] && kill -0 "$(cat "${pid_path}")" 2>/dev/null; then
    echo "[${name}] running pid=$(cat "${pid_path}") url=http://${HEALTH_HOST}:${port}/v1"
    curl -fsS -H "Authorization: Bearer ${VLLM_API_KEY}" "http://${HEALTH_HOST}:${port}/v1/models" || true
    echo
  else
    echo "[${name}] stopped"
  fi
}

case "${COMMAND}" in
  start-one)
    server_modules=()
    SERVER_NAME="${SERVER_NAME:?SERVER_NAME is required for start-one}"
    SERVER_GPUS="${SERVER_GPUS:?SERVER_GPUS is required for start-one}"
    SERVER_PORT="${SERVER_PORT:?SERVER_PORT is required for start-one}"
    SERVER_MODEL="${SERVER_MODEL:?SERVER_MODEL is required for start-one}"
    SERVER_SERVED_MODEL_NAME="${SERVER_SERVED_MODEL_NAME:-${SERVER_NAME}}"
    SERVER_LORAS="${SERVER_LORAS:-}"
    append_lora_modules server_modules "${SERVER_LORAS}"
    start_server "${SERVER_NAME}" "${SERVER_GPUS}" "${SERVER_PORT}" "${SERVER_MODEL}" "${SERVER_SERVED_MODEL_NAME}" "${server_modules[@]}"
    ;;
  stop-one)
    SERVER_NAME="${SERVER_NAME:?SERVER_NAME is required for stop-one}"
    stop_server "${SERVER_NAME}"
    ;;
  status-one)
    SERVER_NAME="${SERVER_NAME:?SERVER_NAME is required for status-one}"
    SERVER_PORT="${SERVER_PORT:?SERVER_PORT is required for status-one}"
    status_server "${SERVER_NAME}" "${SERVER_PORT}"
    ;;
  start)
    teacher_modules=()
    student_modules=()
    append_lora_modules teacher_modules "${TEACHER_LORAS}"
    append_lora_modules teacher_modules "${EXTRA_TEACHER_LORAS}"
    append_lora_modules student_modules "${STUDENT_LORAS}"
    append_lora_modules student_modules "${EXTRA_STUDENT_LORAS}"

    if [[ "${RUN_TEACHER}" == "1" ]]; then
      start_server teacher "${TEACHER_GPUS}" "${TEACHER_PORT}" "${TEACHER_BASE}" "${TEACHER_SERVED_MODEL_NAME}" "${teacher_modules[@]}"
    fi
    if [[ "${RUN_STUDENT}" == "1" ]]; then
      start_server student "${STUDENT_GPUS}" "${STUDENT_PORT}" "${STUDENT_BASE_MODEL}" "${STUDENT_SERVED_MODEL_NAME}" "${student_modules[@]}"
    fi
    ;;
  stop)
    stop_server teacher
    stop_server student
    ;;
  status)
    status_server teacher "${TEACHER_PORT}"
    status_server student "${STUDENT_PORT}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${COMMAND}" >&2
    usage
    exit 1
    ;;
esac
