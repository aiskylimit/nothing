#! /bin/bash

set -euo pipefail

if [[ -n "${RUN_GPUS:-}" ]]; then
  IFS=', ' read -r -a GPUS <<< "${RUN_GPUS}"
else
  GPUS=(0)
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${RUN_MASTER_PORT:-67$(($RANDOM%90+10))}"
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
SCRIPT_NAME="$(basename "${ENTRY_SCRIPT}" .sh)"
SCRIPT_GROUP="$(basename "$(dirname "${ENTRY_SCRIPT}")")"

DATA_DIR="${DATA_DIR:-processed_data/spider_data/llama}"

CKPT="${CKPT:?CKPT must be set by the entry script}"
CKPT_NAME="${CKPT_NAME:?CKPT_NAME must be set by the entry script}"
USE_LORA="${USE_LORA:-0}"

BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-0.0001}"
GRAD_ACC="${GRAD_ACC:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"

# Llama 1B tokenizer stats on benchmarks_2/spider_data/format_data/train.jsonl:
# prompt max = 1496, prompt + response max = 1621.
MAX_LENGTH="${MAX_LENGTH:-1664}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
T_MAX_LENGTH="${T_MAX_LENGTH:-2048}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-1800}"

PEFT_LORA_R="${PEFT_LORA_R:-16}"
PEFT_LORA_ALPHA="${PEFT_LORA_ALPHA:-64}"
PEFT_LORA_DROPOUT="${PEFT_LORA_DROPOUT:-0.1}"

LORA_TAG=""
if [[ "${USE_LORA}" == "1" ]]; then
  LORA_TAG="-lora-${PEFT_LORA_R}-${PEFT_LORA_ALPHA}-${PEFT_LORA_DROPOUT}"
fi

RUN_TAG="e${EPOCHS}-bs${BATCH_SIZE}-lr${LR}-G${GRAD_ACC}-N${GPUS_PER_NODE}-NN${NNODES}${LORA_TAG}"
SAVE_TAG="${SAVE_TAG:-${SCRIPT_GROUP}_${SCRIPT_NAME}_spider_lm_${RUN_TAG}}"
SAVE_PATH="${SAVE_PATH:-${BASE_PATH}/results/llama/${SAVE_TAG}}"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --model-type llama"
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
OPTS+=" --type lm"
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"

if [[ "${USE_LORA}" == "1" ]]; then
  OPTS+=" --peft lora"
  OPTS+=" --peft-lora-r ${PEFT_LORA_R}"
  OPTS+=" --peft-lora-alpha ${PEFT_LORA_ALPHA}"
  OPTS+=" --peft-lora-dropout ${PEFT_LORA_DROPOUT}"
fi

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetuning/finetune.py ${OPTS} $*"

echo "${CMD}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "Model: ${CKPT}"
echo "Data dir: ${DATA_DIR}"
echo "Save path: ${SAVE_PATH}"
echo "LoRA: ${USE_LORA}"
echo "Length config:"
echo "  max length: ${MAX_LENGTH}"
echo "  max prompt length: ${MAX_PROMPT_LENGTH}"
mkdir -p "${SAVE_PATH}"
CODE_BASE=HF ${CMD}
