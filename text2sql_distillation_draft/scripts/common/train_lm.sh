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
MASTER_PORT="${RUN_MASTER_PORT:-66$(($RANDOM % 90 + 10))}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${#GPUS[@]}"

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

BASE_PATH="${BASE_PATH:-.}"
RUN_NAME="${RUN_NAME:?RUN_NAME must be set by the wrapper script}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set by the wrapper script}"
CKPT_NAME="${CKPT_NAME:?CKPT_NAME must be set by the wrapper script}"
MODEL_TYPE="${MODEL_TYPE:?MODEL_TYPE must be set by the wrapper script}"
DATA_DIR="${DATA_DIR:?DATA_DIR must point to a processed mmap data directory, e.g. processed_data/benchmarks/spider_data/qwen}"
USE_LORA="${USE_LORA:-0}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-1}"
LR="${LR:-0.0001}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SEED="${SEED:-42}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
T_MAX_LENGTH="${T_MAX_LENGTH:-2048}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-1800}"

PEFT_LORA_R="${PEFT_LORA_R:-16}"
PEFT_LORA_ALPHA="${PEFT_LORA_ALPHA:-64}"
PEFT_LORA_DROPOUT="${PEFT_LORA_DROPOUT:-0.1}"

LORA_TAG=""
if [[ "${USE_LORA}" =~ ^(1|true|yes|y)$ ]]; then
  LORA_TAG="-lora-${PEFT_LORA_R}-${PEFT_LORA_ALPHA}-${PEFT_LORA_DROPOUT}"
fi

RUN_TAG="e${EPOCHS}-bs${BATCH_SIZE}-lr${LR}-G${GRAD_ACC}-N${GPUS_PER_NODE}-NN${NNODES}${LORA_TAG}"
SAVE_PATH="${SAVE_PATH:-${BASE_PATH}/results/${RUN_NAME}_spider_lm_${RUN_TAG}}"

OPTS=(
  --base-path "${BASE_PATH}"
  --model-path "${MODEL_PATH}"
  --ckpt-name "${CKPT_NAME}"
  --model-type "${MODEL_TYPE}"
  --n-gpu "${GPUS_PER_NODE}"
  --n-nodes "${NNODES}"
  --gradient-checkpointing
  --data-dir "${DATA_DIR}"
  --num-workers 0
  --dev-num -1
  --lr "${LR}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACC}"
  --warmup-iters 0
  --warmup-ratio 0.1
  --lr-decay-style wrmup_cosine
  --weight-decay 1e-2
  --clip-grad 1.0
  --epochs "${EPOCHS}"
  --max-length "${MAX_LENGTH}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --t-max-length "${T_MAX_LENGTH}"
  --t-max-prompt-length "${T_MAX_PROMPT_LENGTH}"
  --do-train
  --do-valid
  --eval-gen
  --save-interval -1
  --eval-interval -1
  --log-interval 20
  --mid-log-num -1
  --save "${SAVE_PATH}"
  --seed "${SEED}"
  --deepspeed
  --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
  --type lm
  --do-sample
  --top-k 0
  --top-p 0.95
  --temperature 0.5
)

if [[ "${USE_LORA}" =~ ^(1|true|yes|y)$ ]]; then
  OPTS+=(
    --peft lora
    --peft-lora-r "${PEFT_LORA_R}"
    --peft-lora-alpha "${PEFT_LORA_ALPHA}"
    --peft-lora-dropout "${PEFT_LORA_DROPOUT}"
  )
fi

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}"

mkdir -p "${SAVE_PATH}"
echo "Run: ${RUN_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_DIR}"
echo "Save: ${SAVE_PATH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "LoRA: ${USE_LORA}"
echo "Training: epochs=${EPOCHS} batch=${BATCH_SIZE} grad_acc=${GRAD_ACC} lr=${LR}"
echo "Lengths: max=${MAX_LENGTH} prompt=${MAX_PROMPT_LENGTH} teacher_max=${T_MAX_LENGTH} teacher_prompt=${T_MAX_PROMPT_LENGTH}"

CODE_BASE=HF torchrun "${DISTRIBUTED_ARGS[@]}" "${BASE_PATH}/finetuning/finetune.py" "${OPTS[@]}" "$@"
