#! /bin/bash

set -euo pipefail

if [[ -n "${RUN_GPUS:-}" ]]; then
  IFS=', ' read -r -a GPUS <<< "${RUN_GPUS}"
else
  GPUS=(0 1)
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=${RUN_MASTER_PORT:-66$(($RANDOM%90+10))}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=.
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
SCRIPT_GROUP="$(basename "$(dirname "${BASH_SOURCE[0]}")")"

DATA_DIR="${DATA_DIR:-processed_data/benchmarks/spider_data/qwen}"

CKPT_NAME="qwen3-0.6B"
CKPT="${CKPT:-Qwen/Qwen3-0.6B}"
TEACHER_CKPT_NAME="qwen3-4B"
TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen3-4B-Instruct-2507}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-hf://distillation-sql/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090}"

BATCH_SIZE=1
LR=0.0001
GRAD_ACC=16
EVAL_BATCH_SIZE=16
EPOCHS=5
KD_RATIO=0.7

MAX_LENGTH="${MAX_LENGTH:-1612}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1479}"

PEFT_LORA_R=16
PEFT_LORA_ALPHA=64
PEFT_LORA_DROPOUT=0.1

GEN_NUM_BEAMS=1
GEN_TOP_P=1.0
INIT_THRESHOLD=0.0
LOSS_EPS=0.1
CAPACITY=1000

RUN_TAG="e${EPOCHS}-bs${BATCH_SIZE}-lr${LR}-G${GRAD_ACC}-N${GPUS_PER_NODE}-NN${NNODES}-kd${KD_RATIO}-lora-${PEFT_LORA_R}-${PEFT_LORA_ALPHA}-${PEFT_LORA_DROPOUT}"
SAVE_TAG="${SAVE_TAG:-${SCRIPT_GROUP}_${SCRIPT_NAME}_spider_adaptive-srkl_${RUN_TAG}}"
SAVE_PATH="${BASE_PATH}/results/qwen3/${SAVE_TAG}"
SEED=42

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
OPTS+=" --model-type qwen"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
# hp
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
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 20"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# type
OPTS+=" --type adaptive-srkl"
# generation
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"
# distillm
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams ${GEN_NUM_BEAMS}"
OPTS+=" --gen-top-p ${GEN_TOP_P}"
OPTS+=" --init-threshold ${INIT_THRESHOLD}"
OPTS+=" --loss-eps ${LOSS_EPS}"
OPTS+=" --capacity ${CAPACITY}"
# lora
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r ${PEFT_LORA_R}"
OPTS+=" --peft-lora-alpha ${PEFT_LORA_ALPHA}"
OPTS+=" --peft-lora-dropout ${PEFT_LORA_DROPOUT}"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetuning/finetune.py ${OPTS} $@"

echo "${CMD}"
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p "${SAVE_PATH}"
CODE_BASE=HF ${CMD}
