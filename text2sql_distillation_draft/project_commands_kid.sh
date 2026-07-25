#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/$(basename "${BASH_SOURCE[0]}")"
cd "${ROOT_DIR}"

MODEL_FAMILY="${MODEL_FAMILY:-both}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  LOG_TO_FILE="${LOG_TO_FILE:-0}"
else
  LOG_TO_FILE="${LOG_TO_FILE:-1}"
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_both_families() {
  local run_ts
  local parent_log_dir
  local qwen_log_dir
  local llama_log_dir
  local qwen_log_file
  local llama_log_file
  local qwen_pid
  local llama_pid
  local failures=0

  run_ts="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
  parent_log_dir="${LOG_DIR:-run_logs/kid/both/${run_ts}}"
  qwen_log_dir="${parent_log_dir}/qwen"
  llama_log_dir="${parent_log_dir}/llama"
  qwen_log_file="${qwen_log_dir}/project_commands_kid.qwen.log"
  llama_log_file="${llama_log_dir}/project_commands_kid.llama.log"

  echo "[kid] mode=both"
  echo "[kid] qwen_gpus=${QWEN_GPUS:-0,1,2,3}, llama_gpus=${LLAMA_GPUS:-4,5,6,7}"
  echo "[kid] batch=${PER_DEVICE_TRAIN_BATCH_SIZE:-8}, grad_acc=${GRAD_ACC:-1}, target_eff=${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
  echo "[kid] infer_seeds=${INFER_SEEDS:-10,42,50,100,1234}"
  echo "[kid] qwen_log=${qwen_log_file}"
  echo "[kid] llama_log=${llama_log_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    (
      unset KID_LOG_ACTIVE RUN_NAME OUTPUT_DIR MODEL_NAME_OR_PATH TEMPLATE LOG_FILE LOG_DIR
      RUN_TS="${run_ts}" \
      LOG_TO_FILE="${LOG_TO_FILE}" \
      MODEL_FAMILY=qwen \
      RUN_GPUS="${QWEN_GPUS:-0,1,2,3}" \
      EVAL_GPUS="${QWEN_EVAL_GPUS:-${QWEN_GPUS:-0,1,2,3}}" \
      PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
      GRAD_ACC="${GRAD_ACC:-1}" \
      TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}" \
      INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}" \
      DRY_RUN="${DRY_RUN}" \
      bash "${SCRIPT_PATH}"
    )
    (
      unset KID_LOG_ACTIVE RUN_NAME OUTPUT_DIR MODEL_NAME_OR_PATH TEMPLATE LOG_FILE LOG_DIR
      RUN_TS="${run_ts}" \
      LOG_TO_FILE="${LOG_TO_FILE}" \
      MODEL_FAMILY=llama \
      RUN_GPUS="${LLAMA_GPUS:-4,5,6,7}" \
      EVAL_GPUS="${LLAMA_EVAL_GPUS:-${LLAMA_GPUS:-4,5,6,7}}" \
      PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
      GRAD_ACC="${GRAD_ACC:-1}" \
      TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}" \
      INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}" \
      DRY_RUN="${DRY_RUN}" \
      bash "${SCRIPT_PATH}"
    )
    return 0
  fi

  if [[ "${SYNC_ENV:-1}" == "1" ]]; then
    echo "[kid] preparing shared env before parallel jobs"
    if [[ "${DEACTIVATE_OLD_ENV:-1}" == "1" ]]; then
      if declare -F deactivate >/dev/null 2>&1; then
        deactivate || true
      fi
      if [[ -n "${CONDA_PREFIX:-}" ]] && command -v conda >/dev/null 2>&1; then
        conda deactivate || true
      fi
    fi
    uv sync
  fi

  mkdir -p "${qwen_log_dir}" "${llama_log_dir}"

  (
    unset KID_LOG_ACTIVE RUN_NAME OUTPUT_DIR MODEL_NAME_OR_PATH TEMPLATE
    RUN_TS="${run_ts}" \
    LOG_TO_FILE=1 \
    LOG_DIR="${qwen_log_dir}" \
    LOG_FILE="${qwen_log_file}" \
    SYNC_ENV=0 \
    MODEL_FAMILY=qwen \
    RUN_GPUS="${QWEN_GPUS:-0,1,2,3}" \
    EVAL_GPUS="${QWEN_EVAL_GPUS:-${QWEN_GPUS:-0,1,2,3}}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
    GRAD_ACC="${GRAD_ACC:-1}" \
    TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}" \
    INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}" \
    bash "${SCRIPT_PATH}"
  ) &
  qwen_pid="$!"

  (
    unset KID_LOG_ACTIVE RUN_NAME OUTPUT_DIR MODEL_NAME_OR_PATH TEMPLATE
    RUN_TS="${run_ts}" \
    LOG_TO_FILE=1 \
    LOG_DIR="${llama_log_dir}" \
    LOG_FILE="${llama_log_file}" \
    SYNC_ENV=0 \
    MODEL_FAMILY=llama \
    RUN_GPUS="${LLAMA_GPUS:-4,5,6,7}" \
    EVAL_GPUS="${LLAMA_EVAL_GPUS:-${LLAMA_GPUS:-4,5,6,7}}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
    GRAD_ACC="${GRAD_ACC:-1}" \
    TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}" \
    INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}" \
    bash "${SCRIPT_PATH}"
  ) &
  llama_pid="$!"

  if ! wait "${qwen_pid}"; then
    echo "[kid][fail] qwen failed; log=${qwen_log_file}" >&2
    failures=$((failures + 1))
  fi
  if ! wait "${llama_pid}"; then
    echo "[kid][fail] llama failed; log=${llama_log_file}" >&2
    failures=$((failures + 1))
  fi

  if [[ "${failures}" -gt 0 ]]; then
    exit 1
  fi
}

case "${MODEL_FAMILY}" in
  both|all|parallel)
    run_both_families
    exit 0
    ;;
esac

if [[ "${LOG_TO_FILE}" == "1" && -z "${KID_LOG_ACTIVE:-}" ]]; then
  RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
  LOG_DIR="${LOG_DIR:-run_logs/kid/${MODEL_FAMILY}/${RUN_TS}}"
  LOG_FILE="${LOG_FILE:-${LOG_DIR}/project_commands_kid.${MODEL_FAMILY}.log}"
  mkdir -p "${LOG_DIR}"
  export LOG_DIR
  export LOG_FILE
  export KID_LOG_ACTIVE=1
  echo "[kid] log_file=${LOG_FILE}"
  exec >> "${LOG_FILE}" 2>&1
fi

echo "[kid] log_file=${LOG_FILE:-<disabled>}"
echo "[kid] dry_run=${DRY_RUN}"

SYNC_ENV="${SYNC_ENV:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[kid][dry-run] skip environment sync/activation"
elif [[ "${SYNC_ENV}" == "1" ]]; then
  if [[ "${DEACTIVATE_OLD_ENV:-1}" == "1" ]]; then
    if declare -F deactivate >/dev/null 2>&1; then
      deactivate || true
    fi
    if [[ -n "${CONDA_PREFIX:-}" ]] && command -v conda >/dev/null 2>&1; then
      conda deactivate || true
    fi
  fi
  uv sync
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

KID_DIR="${ROOT_DIR}/kid/KID-code"
cd "${KID_DIR}"

case "${MODEL_FAMILY}" in
  qwen)
    TRAIN_SCRIPT="scripts/train_kid_spider_qwen3_0.6b_4b.sh"
    DEFAULT_MODEL_TAG="qwen3-0.6b"
    DEFAULT_TEACHER_TAG="qwen3-4b"
    DEFAULT_MODEL_NAME_OR_PATH="Qwen/Qwen3-0.6B"
    DEFAULT_TEMPLATE="qwen3"
    ;;
  llama)
    TRAIN_SCRIPT="scripts/train_kid_spider_llama3_1b_8b.sh"
    DEFAULT_MODEL_TAG="llama3.2-1b-instruct"
    DEFAULT_TEACHER_TAG="llama3.1-8b-instruct"
    DEFAULT_MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1B-Instruct"
    DEFAULT_TEMPLATE="llama3"
    ;;
  *)
    echo "Unsupported MODEL_FAMILY=${MODEL_FAMILY}. Use qwen or llama." >&2
    exit 2
    ;;
esac

RUN_GPUS="${RUN_GPUS:-0,1,2,3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-1}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
LR="${LR:-0.0001}"
EPOCHS="${EPOCHS:-5}"
SAVE_STEPS="${SAVE_STEPS:-1090}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-1536}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-512}"
MASK_RATIO="${MASK_RATIO:-0.2}"
MASK_STRATEGY="${MASK_STRATEGY:-random}"
KL_METHOD="${KL_METHOD:-reverse}"
SAMPLE_SOURCE="${SAMPLE_SOURCE:-mask_student}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET="${LORA_TARGET:-q_proj,v_proj}"
SEED="${SEED:-42}"
DATASET_DIR="${DATASET_DIR:-dbgpt_hub/data/spider_benchmarks_codes}"
DATASET="${DATASET:-example_text2sql_train}"
SPIDER_ROOT="${SPIDER_ROOT:-../../benchmarks/spider_data}"
PREPARE_DATA="${PREPARE_DATA:-1}"
INCLUDE_TRAIN_OTHERS="${INCLUDE_TRAIN_OTHERS:-0}"
TEMPLATE="${TEMPLATE:-${DEFAULT_TEMPLATE}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${DEFAULT_MODEL_NAME_OR_PATH}}"

export RUN_GPUS
export PER_DEVICE_TRAIN_BATCH_SIZE
export GRAD_ACC
export TARGET_EFFECTIVE_BATCH_SIZE
export LR
export EPOCHS
export SAVE_STEPS
export SAVE_TOTAL_LIMIT
export LOGGING_STEPS
export MAX_SOURCE_LENGTH
export MAX_TARGET_LENGTH
export MASK_RATIO
export MASK_STRATEGY
export KL_METHOD
export SAMPLE_SOURCE
export LORA_R
export LORA_ALPHA
export LORA_DROPOUT
export LORA_TARGET
export SEED
export DATASET_DIR
export DATASET
export SPIDER_ROOT
export PREPARE_DATA
export INCLUDE_TRAIN_OTHERS
export TEMPLATE
export MODEL_NAME_OR_PATH
export DRY_RUN

IFS=', ' read -r -a GPU_IDS <<< "${RUN_GPUS}"
NUM_GPUS="${#GPU_IDS[@]}"
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACC * NUM_GPUS))

RUN_NAME="${RUN_NAME:-kid_${DEFAULT_MODEL_TAG}_${DEFAULT_TEACHER_TAG}_spider_e${EPOCHS}-pbs${PER_DEVICE_TRAIN_BATCH_SIZE}-G${GRAD_ACC}-N${NUM_GPUS}-eff${EFFECTIVE_BATCH_SIZE}-${KL_METHOD}-${SAMPLE_SOURCE}-${MASK_STRATEGY}${MASK_RATIO}-lora-${LORA_R}-${LORA_ALPHA}-${LORA_DROPOUT}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-dbgpt_hub/output/adapter_kd/spider/${RUN_NAME}}"
export RUN_NAME
export OUTPUT_DIR

echo "[kid] model_family=${MODEL_FAMILY}"
echo "[kid] gpus=${RUN_GPUS}, effective_batch=${EFFECTIVE_BATCH_SIZE}"
echo "[kid] run_name=${RUN_NAME}"
echo "[kid] output_dir=${OUTPUT_DIR}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  bash "${TRAIN_SCRIPT}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  EVAL_GPUS="${EVAL_GPUS:-${RUN_GPUS}}"
  IFS=', ' read -r -a EVAL_GPU_IDS <<< "${EVAL_GPUS}"
  EVAL_SPLITS="${#EVAL_GPU_IDS[@]}"
  EVAL_CHECKPOINTS="${EVAL_CHECKPOINTS:-checkpoint-${SAVE_STEPS},checkpoint_lastest}"
  INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"
  PREDICT_INPUT_FILENAME="${PREDICT_INPUT_FILENAME:-${DATASET_DIR}/example_text2sql_dev.json}"

  if [[ "${EVAL_SPLITS}" -lt 1 ]]; then
    echo "EVAL_GPUS must contain at least one GPU id." >&2
    exit 2
  fi

  for checkpoint_step in ${EVAL_CHECKPOINTS//,/ }; do
    checkpoint_dir="dbgpt_hub/output/adapter_kd/spider/${RUN_NAME}/${checkpoint_step}"
    if [[ "${checkpoint_step}" == "checkpoint_lastest" ]]; then
      checkpoint_dir="dbgpt_hub/output/adapter_kd/spider/${RUN_NAME}"
    fi

    for infer_seed in ${INFER_SEEDS//,/ }; do
      output_path="dbgpt_hub/output/adapter_kd/spider/${RUN_NAME}/preds/${checkpoint_step}/seed${infer_seed}"

      echo "[kid] infer/eval ${checkpoint_step} seed=${infer_seed} on gpus=${EVAL_GPUS}"
      if [[ "${DRY_RUN}" == "1" ]]; then
        print_command mkdir -p "${output_path}"
      else
        mkdir -p "${output_path}"
      fi

      for idx in "${!EVAL_GPU_IDS[@]}"; do
        cuda="${EVAL_GPU_IDS[$idx]}"
        EVAL_CMD=(
          python dbgpt_hub/predict/predict.py
          --model_name_or_path "${MODEL_NAME_OR_PATH}"
          --template "${TEMPLATE}"
          --split_part "${idx}:${EVAL_SPLITS}"
          --finetuning_type lora
          --predicted_input_filename "${PREDICT_INPUT_FILENAME}"
          --checkpoint_dir "${checkpoint_dir}"
          --predicted_out_filename "${output_path}/pred_codes-${idx}:${EVAL_SPLITS}.sql"
        )
        if [[ "${DRY_RUN}" == "1" ]]; then
          printf 'PYTHONHASHSEED=%q KID_INFER_SEED=%q CUDA_VISIBLE_DEVICES=%q ' "${infer_seed}" "${infer_seed}" "${cuda}"
          print_command "${EVAL_CMD[@]}"
        else
          PYTHONHASHSEED="${infer_seed}" KID_INFER_SEED="${infer_seed}" CUDA_VISIBLE_DEVICES="${cuda}" "${EVAL_CMD[@]}" &
        fi
      done

      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "cat ${output_path}/pred_codes-*.sql > ${output_path}/pred_codes.sql"
        echo "rm -f ${output_path}/pred_codes-*.sql"
        echo "python dbgpt_hub/eval/evaluation.py --plug_value --input ${output_path}/pred_codes.sql > ${output_path}/pred_codes.result"
      else
        wait
        cat "${output_path}"/pred_codes-*.sql > "${output_path}/pred_codes.sql"
        rm -f "${output_path}"/pred_codes-*.sql
        python dbgpt_hub/eval/evaluation.py --plug_value --input "${output_path}/pred_codes.sql" \
          > "${output_path}/pred_codes.result"
      fi
    done
  done
fi
