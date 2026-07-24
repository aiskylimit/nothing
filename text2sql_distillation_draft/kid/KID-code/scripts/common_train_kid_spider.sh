#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASE_DIR}"

: "${MODEL_NAME_OR_PATH:?MODEL_NAME_OR_PATH must be set by the entry script}"
: "${TEACHER_MODEL_PATH:?TEACHER_MODEL_PATH must be set by the entry script}"
: "${MODEL_TAG:?MODEL_TAG must be set by the entry script}"
: "${TEACHER_TAG:?TEACHER_TAG must be set by the entry script}"
: "${TEMPLATE:?TEMPLATE must be set by the entry script}"

RUN_GPUS="${RUN_GPUS:-0}"
IFS=', ' read -r -a GPUS <<< "${RUN_GPUS}"
NUM_GPUS="${#GPUS[@]}"
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
export CUDA_VISIBLE_DEVICES

MASTER_PORT="${RUN_MASTER_PORT:-12$(($RANDOM%900+100))}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-32}"
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRAD_ACC * NUM_GPUS))

if [[ "${EFFECTIVE_BATCH_SIZE}" -ne "${TARGET_EFFECTIVE_BATCH_SIZE}" && "${ALLOW_EFFECTIVE_BATCH_MISMATCH:-0}" != "1" ]]; then
  echo "effective batch mismatch: ${EFFECTIVE_BATCH_SIZE} != ${TARGET_EFFECTIVE_BATCH_SIZE}" >&2
  echo "Use PER_DEVICE_TRAIN_BATCH_SIZE, GRAD_ACC, RUN_GPUS to make per_device * gradacc * num_gpus = ${TARGET_EFFECTIVE_BATCH_SIZE}." >&2
  exit 2
fi

DATASET_DIR="${DATASET_DIR:-dbgpt_hub/data/spider_benchmarks_codes}"
DATASET="${DATASET:-example_text2sql_train}"
SPIDER_ROOT="${SPIDER_ROOT:-../../benchmarks/spider_data}"
PREPARE_DATA="${PREPARE_DATA:-1}"
INCLUDE_TRAIN_OTHERS="${INCLUDE_TRAIN_OTHERS:-0}"

if [[ "${PREPARE_DATA}" == "1" ]]; then
  PREPARE_ARGS=(--spider-root "${SPIDER_ROOT}" --output-dir "${DATASET_DIR}")
  if [[ "${INCLUDE_TRAIN_OTHERS}" == "1" ]]; then
    PREPARE_ARGS+=(--include-train-others)
  fi
  python scripts/prepare_spider_benchmarks_data.py "${PREPARE_ARGS[@]}"
fi

if [[ ! -s "${DATASET_DIR}/dataset_info.json" || ! -s "${DATASET_DIR}/example_text2sql_train.json" ]]; then
  echo "Missing prepared KID Spider data under ${DATASET_DIR}" >&2
  echo "Run: python scripts/prepare_spider_benchmarks_data.py --spider-root ${SPIDER_ROOT} --output-dir ${DATASET_DIR}" >&2
  exit 2
fi

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

RUN_NAME="${RUN_NAME:-kid_${MODEL_TAG}_${TEACHER_TAG}_spider_e${EPOCHS}-pbs${PER_DEVICE_TRAIN_BATCH_SIZE}-G${GRAD_ACC}-N${NUM_GPUS}-eff${EFFECTIVE_BATCH_SIZE}-${KL_METHOD}-${SAMPLE_SOURCE}-${MASK_STRATEGY}${MASK_RATIO}-lora-${LORA_R}-${LORA_ALPHA}-${LORA_DROPOUT}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-dbgpt_hub/output/adapter_kd/spider/${RUN_NAME}}"

CMD=(
  deepspeed --num_gpus "${NUM_GPUS}" --master_port "${MASTER_PORT}" dbgpt_hub/train/sft_train.py
  --deepspeed dbgpt_hub/configs/ds_config.json
  --use_kd
  --teacher_model_path "${TEACHER_MODEL_PATH}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --do_train
  --dataset "${DATASET}"
  --dataset_dir "${DATASET_DIR}"
  --kl_method "${KL_METHOD}"
  --sample_source "${SAMPLE_SOURCE}"
  --mask_strategy "${MASK_STRATEGY}"
  --mask_ratio "${MASK_RATIO}"
  --max_source_length "${MAX_SOURCE_LENGTH}"
  --max_target_length "${MAX_TARGET_LENGTH}"
  --template "${TEMPLATE}"
  --finetuning_type lora
  --lora_rank "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_target "${LORA_TARGET}"
  --output_dir "${OUTPUT_DIR}"
  --overwrite_cache
  --overwrite_output_dir
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  --lr_scheduler_type cosine
  --logging_steps "${LOGGING_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --learning_rate "${LR}"
  --num_train_epochs "${EPOCHS}"
  --seed "${SEED}"
  --plot_loss
  --bf16
)

if [[ -n "${TEACHER_PEFT_PATH:-}" ]]; then
  CMD+=(--teacher_peft_path "${TEACHER_PEFT_PATH}")
fi

echo "KID Spider training"
echo "  student: ${MODEL_NAME_OR_PATH}"
echo "  teacher: ${TEACHER_MODEL_PATH}"
echo "  teacher peft: ${TEACHER_PEFT_PATH:-<none>}"
echo "  template: ${TEMPLATE}"
echo "  data: ${DATASET_DIR}/${DATASET}"
echo "  gpus: ${CUDA_VISIBLE_DEVICES}"
echo "  per-device batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  grad acc: ${GRAD_ACC}"
echo "  effective batch: ${EFFECTIVE_BATCH_SIZE}"
echo "  output: ${OUTPUT_DIR}"
echo "${CMD[*]}"

mkdir -p "${OUTPUT_DIR}"
"${CMD[@]}"
