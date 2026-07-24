#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

SYNC_ENV="${SYNC_ENV:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
UNZIP_DATA="${UNZIP_DATA:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ "${SYNC_ENV}" == "1" ]]; then
  uv sync
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  hf download Dream-AI-HUST/sql_benchmarks \
    --repo-type dataset \
    --local-dir .
fi

if [[ "${UNZIP_DATA}" == "1" ]]; then
  unzip -o benchmarks.zip
  unzip -o data.zip
fi

KID_DIR="${ROOT_DIR}/kid/KID-code"
cd "${KID_DIR}"

MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
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
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-4}"
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
  PREDICT_INPUT_FILENAME="${PREDICT_INPUT_FILENAME:-${DATASET_DIR}/example_text2sql_dev.json}"

  if [[ "${EVAL_SPLITS}" -lt 1 ]]; then
    echo "EVAL_GPUS must contain at least one GPU id." >&2
    exit 2
  fi

  for checkpoint_step in ${EVAL_CHECKPOINTS//,/ }; do
    output_path="dbgpt_hub/output/adapter_kd/spider/${RUN_NAME}/preds/${checkpoint_step}"
    checkpoint_dir="dbgpt_hub/output/adapter_kd/spider/${RUN_NAME}/${checkpoint_step}"
    if [[ "${checkpoint_step}" == "checkpoint_lastest" ]]; then
      checkpoint_dir="dbgpt_hub/output/adapter_kd/spider/${RUN_NAME}"
    fi
    mkdir -p "${output_path}"

    echo "[kid] infer/eval ${checkpoint_step} on gpus=${EVAL_GPUS}"
    for idx in "${!EVAL_GPU_IDS[@]}"; do
      cuda="${EVAL_GPU_IDS[$idx]}"
      CUDA_VISIBLE_DEVICES="${cuda}" python dbgpt_hub/predict/predict.py \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --template "${TEMPLATE}" \
        --split_part "${idx}:${EVAL_SPLITS}" \
        --finetuning_type lora \
        --predicted_input_filename "${PREDICT_INPUT_FILENAME}" \
        --checkpoint_dir "${checkpoint_dir}" \
        --predicted_out_filename "${output_path}/pred_codes-${idx}:${EVAL_SPLITS}.sql" &
    done
    wait

    cat "${output_path}"/pred_codes-*.sql > "${output_path}/pred_codes.sql"
    rm -f "${output_path}"/pred_codes-*.sql
    python dbgpt_hub/eval/evaluation.py --plug_value --input "${output_path}/pred_codes.sql" \
      > "${output_path}/pred_codes.result"
  done
fi
