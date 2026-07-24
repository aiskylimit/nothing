#! /usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

RUN_ROOT="${RUN_ROOT:-results/llama_synid_sql}"
RUN_GLOB="${RUN_GLOB:-synid_ce_keywords_weight_lora_218_train_g*_spider_synid_*}"
OUT_ROOT="${OUT_ROOT:-results/infer/llama_synid_sql/latest_ckpt}"
LOG_DIR="${LOG_DIR:-run_logs/llama_synid_infer/$(date +%Y%m%d_%H%M%S)}"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
INFER_SCRIPT="${INFER_SCRIPT:-scripts/qwen_updated_3_218/synid_ce_keywords_weight_lora/infer_multiseed.py}"
INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"
BENCHMARKS="${BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
SPLIT="${SPLIT:-test}"
DB="${DB:-full}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LENGTH="${MAX_LENGTH:-auto}"
FLUSH_EVERY="${FLUSH_EVERY:-20}"
DEVICE="${DEVICE:-cuda}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
RUN_MODE="${RUN_MODE:-parallel}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"

latest_numeric_ckpt() {
  local run_dir="$1"
  local best_step="-1"
  local best_path=""
  local ckpt_path
  local ckpt_name
  local ckpt_step

  for ckpt_path in "${run_dir}"/*; do
    [[ -d "${ckpt_path}" ]] || continue
    ckpt_name="$(basename "${ckpt_path}")"
    [[ "${ckpt_name}" =~ ^[0-9]+$ ]] || continue
    ckpt_step=$((10#${ckpt_name}))
    if (( ckpt_step > best_step )); then
      best_step="${ckpt_step}"
      best_path="${ckpt_path}"
    fi
  done

  [[ -n "${best_path}" ]] && printf '%s\n' "${best_path}"
}

infer_max_length_for() {
  local benchmark="$1"
  local split="$2"

  if [[ "${MAX_LENGTH}" != "auto" ]]; then
    printf '%s\n' "${MAX_LENGTH}"
    return
  fi

  case "${benchmark}:${split}" in
    spider_data:train) printf '1612\n' ;;
    spider_data:test) printf '856\n' ;;
    spider_syn:test) printf '756\n' ;;
    spider_realistic:test) printf '755\n' ;;
    spider_dk:test) printf '663\n' ;;
    *) printf '856\n' ;;
  esac
}

run_one() {
  local run_dir="$1"
  local gpu="$2"
  local run_name
  local ckpt
  local ckpt_step
  local benchmark
  local output_dir
  local output_path
  local log_path
  local max_length

  run_name="$(basename "${run_dir}")"
  ckpt="$(latest_numeric_ckpt "${run_dir}" || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "[infer-skip] no numeric checkpoint under ${run_dir}" >&2
    return 1
  fi

  ckpt_step="$(basename "${ckpt}")"

  IFS=',' read -r -a benchmark_list <<< "${BENCHMARKS}"
  for benchmark in "${benchmark_list[@]}"; do
    benchmark="${benchmark// /}"
    [[ -n "${benchmark}" ]] || continue

    output_dir="${OUT_ROOT}/${benchmark}"
    output_path="${output_dir}/${run_name}__ckpt${ckpt_step}__${SPLIT}__${DB}_sql_result.json"
    log_path="${LOG_DIR}/${run_name}__ckpt${ckpt_step}__${benchmark}__${SPLIT}__${DB}.log"
    max_length="$(infer_max_length_for "${benchmark}" "${SPLIT}")"

    mkdir -p "${output_dir}" "${LOG_DIR}"

    echo "[infer] gpu=${gpu} run=${run_name} ckpt=${ckpt_step} benchmark=${benchmark} seeds=${INFER_SEEDS} batch=${BATCH_SIZE}"
    (
      echo "run_dir=${run_dir}"
      echo "ckpt=${ckpt}"
      echo "output=${output_path}"
      echo "seeds=${INFER_SEEDS}"
      CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONPATH="${ROOT_DIR}" \
        INFER_SEEDS="${INFER_SEEDS}" \
        SKIP_EXISTING="${SKIP_EXISTING}" \
        INFER_CKPT_PATH="${ckpt}" \
        INFER_CKPT_STEP="${ckpt_step}" \
        python "${INFER_SCRIPT}" \
        --benchmark "${benchmark}" \
        --split "${SPLIT}" \
        --db "${DB}" \
        --model "${MODEL}" \
        --ckpt_path "${ckpt}" \
        --device "${DEVICE}" \
        --batch-size "${BATCH_SIZE}" \
        --max-length "${max_length}" \
        --flush-every "${FLUSH_EVERY}" \
        --output_path "${output_path}"
    ) > "${log_path}" 2>&1
    echo "[infer-done] ${output_path}"
  done
}

if [[ "${RUN_MODE}" != "parallel" && "${RUN_MODE}" != "sequential" ]]; then
  echo "RUN_MODE must be parallel or sequential, got: ${RUN_MODE}" >&2
  exit 2
fi

IFS=',' read -r -a gpus <<< "${GPU_LIST}"
if [[ "${#gpus[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

shopt -s nullglob
run_dirs=("${RUN_ROOT}"/${RUN_GLOB})
shopt -u nullglob

if [[ "${#run_dirs[@]}" -eq 0 ]]; then
  echo "No run dirs matched: ${RUN_ROOT}/${RUN_GLOB}" >&2
  exit 1
fi

echo "[infer] run_dirs=${#run_dirs[@]}"
echo "[infer] gpus=${GPU_LIST}, mode=${RUN_MODE}"
echo "[infer] model=${MODEL}"
echo "[infer] script=${INFER_SCRIPT}"
echo "[infer] benchmarks=${BENCHMARKS}, split=${SPLIT}, db=${DB}"
echo "[infer] seeds=${INFER_SEEDS}, batch_size=${BATCH_SIZE}, max_length=${MAX_LENGTH}"
echo "[infer] output=${OUT_ROOT}"
echo "[infer] logs=${LOG_DIR}"

if [[ "${RUN_MODE}" == "sequential" ]]; then
  gpu="${gpus[0]}"
  for run_dir in "${run_dirs[@]}"; do
    run_one "${run_dir}" "${gpu}"
  done
else
  pids=()
  for idx in "${!run_dirs[@]}"; do
    gpu="${gpus[$((idx % ${#gpus[@]}))]}"
    run_one "${run_dirs[$idx]}" "${gpu}" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  exit "${status}"
fi
