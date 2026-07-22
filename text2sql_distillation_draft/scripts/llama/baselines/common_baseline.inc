#! /usr/bin/env bash

set -euo pipefail

if [[ -n "${RUN_GPUS:-}" ]]; then
  IFS=', ' read -r -a GPUS <<< "${RUN_GPUS}"
else
  GPUS=(0 1)
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${RUN_MASTER_PORT:-68$(($RANDOM%90+10))}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${#GPUS[@]}"

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
                  --nnodes ${NNODES} \
                  --node_rank ${NODE_RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

BASE_PATH="${BASE_PATH:-.}"
ENTRY_SCRIPT="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
SCRIPT_DIR="$(cd "$(dirname "${ENTRY_SCRIPT}")" && pwd)"
SCRIPT_NAME="$(basename "${ENTRY_SCRIPT}" .sh)"
SCRIPT_GROUP="$(basename "${SCRIPT_DIR}")"

DATA_DIR="${DATA_DIR:-processed_data/spider_data/llama}"

CKPT_NAME="${CKPT_NAME:-llama3.2-1b-instruct}"
CKPT="${CKPT:-meta-llama/Llama-3.2-1B-Instruct}"
TEACHER_CKPT_NAME="${TEACHER_CKPT_NAME:-llama3.1-8b-instruct}"
TEACHER_CKPT="${TEACHER_CKPT:-meta-llama/Llama-3.1-8B-Instruct}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-hf://distillation-sql/llama_spider/llama/sft_sft_llama3_8b_lora_spider_lm_e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/1090}"

BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-0.0001}"
GRAD_ACC="${GRAD_ACC:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
KD_RATIO="${KD_RATIO:-0.7}"
SEED="${SEED:-42}"

# Llama-tokenized Spider stats from scripts/llama/sft.
MAX_LENGTH="${MAX_LENGTH:-1664}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
T_MAX_LENGTH="${T_MAX_LENGTH:-2048}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-1800}"

PEFT_LORA_R="${PEFT_LORA_R:-16}"
PEFT_LORA_ALPHA="${PEFT_LORA_ALPHA:-64}"
PEFT_LORA_DROPOUT="${PEFT_LORA_DROPOUT:-0.1}"

GEN_NUM_BEAMS="${GEN_NUM_BEAMS:-1}"
GEN_TOP_P="${GEN_TOP_P:-1.0}"
INIT_THRESHOLD="${INIT_THRESHOLD:-0.0}"
LOSS_EPS="${LOSS_EPS:-0.1}"
CAPACITY="${CAPACITY:-1000}"

ENABLE_AMID="${ENABLE_AMID:-0}"
ENABLE_FDD="${ENABLE_FDD:-0}"
BASELINE_KIND="${BASELINE_KIND:-${SCRIPT_NAME#train_}}"

if [[ "${ENABLE_FDD}" == "1" ]]; then
  DISTILL_TYPE="${FDD_DISTILL_TYPE:-sfkl}"

  if [[ -n "${STUDENT_LAYER_MAPPING_OVERRIDE:-}" ]]; then
    read -r -a STUDENT_LAYER_MAPPING <<< "${STUDENT_LAYER_MAPPING_OVERRIDE}"
  else
    STUDENT_LAYER_MAPPING=(13 16)
  fi

  if [[ -n "${TEACHER_LAYER_MAPPING_OVERRIDE:-}" ]]; then
    read -r -a TEACHER_LAYER_MAPPING <<< "${TEACHER_LAYER_MAPPING_OVERRIDE}"
  else
    TEACHER_LAYER_MAPPING=(29 32)
  fi

  if [[ "${#STUDENT_LAYER_MAPPING[@]}" -lt 2 || "${#TEACHER_LAYER_MAPPING[@]}" -lt 2 ]]; then
    echo "FDD needs at least two student/teacher layer pairs." >&2
    exit 2
  fi
  if [[ "${#STUDENT_LAYER_MAPPING[@]}" -ne "${#TEACHER_LAYER_MAPPING[@]}" ]]; then
    echo "FDD layer mapping length mismatch." >&2
    exit 2
  fi

  STUDENT_LAYERS_TAG="$(IFS=-; echo "${STUDENT_LAYER_MAPPING[*]}")"
  TEACHER_LAYERS_TAG="$(IFS=-; echo "${TEACHER_LAYER_MAPPING[*]}")"
  METHOD_TAG="fdd_${DISTILL_TYPE}_t${TEACHER_LAYERS_TAG}_s${STUDENT_LAYERS_TAG}"
  TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${BASE_PATH}/finetuning/fdd_finetune.py}"
else
  DISTILL_TYPE="${DISTILL_TYPE:?DISTILL_TYPE must be set by the entry script}"
  METHOD_TAG="${BASELINE_KIND}_${DISTILL_TYPE}"
  TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-${BASE_PATH}/finetuning/finetune.py}"
fi

if [[ "${ENABLE_AMID}" == "1" ]]; then
  AMID_DIV_NAME="${AMID_DIV_NAME:-ab}"
  AMID_DIV_ORDER="${AMID_DIV_ORDER:-pr}"
  AMID_ALPHA="${AMID_ALPHA:-0.5}"
  AMID_LAM="${AMID_LAM:-0.5}"
  METHOD_TAG="amid_${DISTILL_TYPE}_${AMID_DIV_NAME}_${AMID_DIV_ORDER}_${AMID_ALPHA}_${AMID_LAM}"
fi

RUN_TAG="e${EPOCHS}-bs${BATCH_SIZE}-lr${LR}-G${GRAD_ACC}-N${GPUS_PER_NODE}-NN${NNODES}-kd${KD_RATIO}-lora-${PEFT_LORA_R}-${PEFT_LORA_ALPHA}-${PEFT_LORA_DROPOUT}"
SAVE_TAG="${SAVE_TAG:-${SCRIPT_GROUP}_${SCRIPT_NAME}_spider_${METHOD_TAG}_${RUN_TAG}}"
SAVE_PATH="${SAVE_PATH:-${BASE_PATH}/results/llama_baselines/${SAVE_TAG}}"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
OPTS+=" --model-type llama"
OPTS+=" --teacher-model-type llama"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --gradient-checkpointing"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --warmup-ratio 0.1"
OPTS+=" --lr-decay-style wrmup_cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio ${KD_RATIO}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
OPTS+=" --t-max-length ${T_MAX_LENGTH}"
OPTS+=" --t-max-prompt-length ${T_MAX_PROMPT_LENGTH}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 20"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
OPTS+=" --type ${DISTILL_TYPE}"
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams ${GEN_NUM_BEAMS}"
OPTS+=" --gen-top-p ${GEN_TOP_P}"
OPTS+=" --init-threshold ${INIT_THRESHOLD}"
OPTS+=" --loss-eps ${LOSS_EPS}"
OPTS+=" --capacity ${CAPACITY}"

if [[ "${ENABLE_AMID}" == "1" ]]; then
  OPTS+=" --amid-div-name ${AMID_DIV_NAME}"
  OPTS+=" --amid-div-order ${AMID_DIV_ORDER}"
  OPTS+=" --amid-alpha ${AMID_ALPHA}"
  OPTS+=" --amid-lam ${AMID_LAM}"
fi

if [[ "${ENABLE_FDD}" == "1" ]]; then
  OPTS+=" --teacher_layer_mapping ${TEACHER_LAYER_MAPPING[*]}"
  OPTS+=" --student_layer_mapping ${STUDENT_LAYER_MAPPING[*]}"
  if [[ -n "${FDD_WEIGHT:-}" ]]; then
    OPTS+=" --fdd-weight ${FDD_WEIGHT}"
  fi
fi

OPTS+=" --peft lora"
OPTS+=" --peft-lora-r ${PEFT_LORA_R}"
OPTS+=" --peft-lora-alpha ${PEFT_LORA_ALPHA}"
OPTS+=" --peft-lora-dropout ${PEFT_LORA_DROPOUT}"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${TRAIN_ENTRYPOINT} ${OPTS} $*"

echo "${CMD}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "Student: ${CKPT}"
echo "Teacher: ${TEACHER_CKPT}"
echo "Teacher LoRA: ${TEACHER_PEFT_PATH}"
echo "Data dir: ${DATA_DIR}"
echo "Save path: ${SAVE_PATH}"
echo "Baseline: ${METHOD_TAG}"
echo "Length config:"
echo "  max length: ${MAX_LENGTH}"
echo "  max prompt length: ${MAX_PROMPT_LENGTH}"
mkdir -p "${SAVE_PATH}"
CODE_BASE=HF ${CMD}
