#! /usr/bin/env bash

set -euo pipefail

ENTRY_SCRIPT="${BASH_SOURCE[1]}"
SCRIPT_DIR="$(cd "$(dirname "${ENTRY_SCRIPT}")" && pwd)"
SCRIPT_GROUP="$(basename "${SCRIPT_DIR}")"
SCRIPT_NAME="$(basename "${ENTRY_SCRIPT}" .sh)"

if [[ -n "${RUN_GPUS:-}" ]]; then
  IFS=', ' read -r -a GPUS <<< "${RUN_GPUS}"
else
  GPUS=(0 1)
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${RUN_MASTER_PORT:-66$(($RANDOM%90+10))}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${#GPUS[@]}"

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
                  --nnodes ${NNODES} \
                  --node_rank ${NODE_RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

BASE_PATH="${BASE_PATH:-.}"
DATA_DIR="${DATA_DIR:-processed_data/benchmarks/spider_data/qwen}"

CKPT_NAME="${CKPT_NAME:-qwen2.5-0.5B-Instruct}"
CKPT="${CKPT:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER_CKPT_NAME="${TEACHER_CKPT_NAME:-qwen3-4B}"
TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen3-4B-Instruct-2507}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-hf://Dream-AI-HUST/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090}"

KD_TYPE="${KD_TYPE:-${SCRIPT_GROUP}}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-0.0001}"
GRAD_ACC="${GRAD_ACC:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
KD_RATIO="${KD_RATIO:-0.7}"

MAX_LENGTH="${MAX_LENGTH:-1612}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1479}"

PEFT_LORA_R="${PEFT_LORA_R:-16}"
PEFT_LORA_ALPHA="${PEFT_LORA_ALPHA:-64}"
PEFT_LORA_DROPOUT="${PEFT_LORA_DROPOUT:-0.1}"

GEN_NUM_BEAMS="${GEN_NUM_BEAMS:-1}"
GEN_TOP_P="${GEN_TOP_P:-1.0}"
INIT_THRESHOLD="${INIT_THRESHOLD:-0.0}"
LOSS_EPS="${LOSS_EPS:-0.1}"
CAPACITY="${CAPACITY:-1000}"

TRAINER="${TRAINER:-finetuning/finetune.py}"
SAVE_METHOD_TAG="${SAVE_METHOD_TAG:-${KD_TYPE}}"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
OPTS+=" --model-type qwen"
OPTS+=" --teacher-model-type qwen"
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
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 20"
OPTS+=" --mid-log-num -1"
OPTS+=" --seed ${SEED:-42}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
OPTS+=" --type ${KD_TYPE}"
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

if [[ "${ADD_AMID_ARGS:-false}" == "true" ]]; then
  AMID_DIV_NAME="${AMID_DIV_NAME:-ab}"
  AMID_DIV_ORDER="${AMID_DIV_ORDER:-pr}"
  AMID_ALPHA="${AMID_ALPHA:-0.5}"
  AMID_LAM="${AMID_LAM:-0.5}"
  SAVE_METHOD_TAG="${SAVE_METHOD_TAG}_${AMID_DIV_NAME}_${AMID_DIV_ORDER}_${AMID_ALPHA}_${AMID_LAM}"
  OPTS+=" --amid-div-name ${AMID_DIV_NAME}"
  OPTS+=" --amid-div-order ${AMID_DIV_ORDER}"
  OPTS+=" --amid-alpha ${AMID_ALPHA}"
  OPTS+=" --amid-lam ${AMID_LAM}"
fi

if [[ "${ADD_FDD_ARGS:-false}" == "true" ]]; then
  TEACHER_LAYER_MAPPING=(${TEACHER_LAYER_MAPPING:-33 36})
  STUDENT_LAYER_MAPPING=(${STUDENT_LAYER_MAPPING:-21 24})
  TEACHER_LAYERS_TAG="$(IFS=-; echo "${TEACHER_LAYER_MAPPING[*]}")"
  STUDENT_LAYERS_TAG="$(IFS=-; echo "${STUDENT_LAYER_MAPPING[*]}")"
  SAVE_METHOD_TAG="${SAVE_METHOD_TAG}_t${TEACHER_LAYERS_TAG}_s${STUDENT_LAYERS_TAG}"
  OPTS+=" --teacher_layer_mapping ${TEACHER_LAYER_MAPPING[*]}"
  OPTS+=" --student_layer_mapping ${STUDENT_LAYER_MAPPING[*]}"
fi

OPTS+=" --peft lora"
OPTS+=" --peft-lora-r ${PEFT_LORA_R}"
OPTS+=" --peft-lora-alpha ${PEFT_LORA_ALPHA}"
OPTS+=" --peft-lora-dropout ${PEFT_LORA_DROPOUT}"

RUN_TAG="e${EPOCHS}-bs${BATCH_SIZE}-lr${LR}-G${GRAD_ACC}-N${GPUS_PER_NODE}-NN${NNODES}-kd${KD_RATIO}-lora-${PEFT_LORA_R}-${PEFT_LORA_ALPHA}-${PEFT_LORA_DROPOUT}"
SAVE_TAG="${SAVE_TAG:-qwen2.5_${SCRIPT_GROUP}_${SCRIPT_NAME}_spider_${SAVE_METHOD_TAG}_${RUN_TAG}}"
SAVE_PATH="${SAVE_PATH:-${BASE_PATH}/results/qwen2.5/${SAVE_TAG}}"
OPTS+=" --save ${SAVE_PATH}"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/${TRAINER} ${OPTS} $*"

echo "${CMD}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "Student: ${CKPT}"
echo "Teacher: ${TEACHER_CKPT}"
echo "Data dir: ${DATA_DIR}"
echo "KD type: ${KD_TYPE}"
echo "Save path: ${SAVE_PATH}"
mkdir -p "${SAVE_PATH}"
CODE_BASE=HF ${CMD}
