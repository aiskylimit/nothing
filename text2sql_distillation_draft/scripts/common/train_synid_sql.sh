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
MASTER_PORT="${RUN_MASTER_PORT:-67$(($RANDOM % 90 + 10))}"
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
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:?TEACHER_MODEL_PATH must be set by the wrapper script}"
TEACHER_CKPT_NAME="${TEACHER_CKPT_NAME:?TEACHER_CKPT_NAME must be set by the wrapper script}"
TEACHER_MODEL_TYPE="${TEACHER_MODEL_TYPE:-${MODEL_TYPE}}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:?TEACHER_PEFT_PATH must point to the custom teacher LoRA adapter for this run}"
DATA_DIR="${DATA_DIR:?DATA_DIR must point to a SynID processed mmap data directory, e.g. processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen}"

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

KD_TYPE="${KD_TYPE:-synid}"
KD_RATIO="${KD_RATIO:-0.7}"
SYNID_ALPHA="${SYNID_ALPHA:-0.3}"
SYNID_BETA="${SYNID_BETA:-0.3}"
SYNID_KD_LOSS="${SYNID_KD_LOSS:-csd}"
SYNID_POOL_TAU="${SYNID_POOL_TAU:-5.0}"
SYNID_CONTRASTIVE_TAU="${SYNID_CONTRASTIVE_TAU:-0.05}"
SYNID_SYNTAX_LAMBDA="${SYNID_SYNTAX_LAMBDA:-2.0}"
SYNID_POOLING="${SYNID_POOLING:-sc}"
SYNID_USE_SYNTAX_WEIGHTS="${SYNID_USE_SYNTAX_WEIGHTS:-true}"
SYNID_USE_CON1="${SYNID_USE_CON1:-true}"
SYNID_USE_CON2="${SYNID_USE_CON2:-true}"
SYNID_USE_PRIVILEGED_TEACHER_INPUT="${SYNID_USE_PRIVILEGED_TEACHER_INPUT:-true}"
SYNID_STUDENT_LAYERS="${SYNID_STUDENT_LAYERS:--1}"
SYNID_TEACHER_LAYERS="${SYNID_TEACHER_LAYERS:--1}"
SYNID_LAYER_CONFIG="${SYNID_LAYER_CONFIG:-last_layer}"

if [[ "${SYNID_USE_PRIVILEGED_TEACHER_INPUT}" =~ ^(1|true|yes|y)$ ]]; then
  if [[ ! -f "${DATA_DIR}/teacher_train_0.bin" && -f "${DATA_DIR}/${MODEL_TYPE}/teacher_train_0.bin" ]]; then
    DATA_DIR="${DATA_DIR}/${MODEL_TYPE}"
  fi

  if [[ ! -f "${DATA_DIR}/teacher_train_0.bin" || ! -f "${DATA_DIR}/teacher_train_0.idx" ]]; then
    echo "Missing SynID teacher mmap files in DATA_DIR=${DATA_DIR}" >&2
    echo "Expected: ${DATA_DIR}/teacher_train_0.bin and ${DATA_DIR}/teacher_train_0.idx" >&2
    echo "For Qwen, expected data dir is: processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen" >&2
    echo "Unset stale DATA_DIR or pass DATA_DIR=<correct path> before running this script." >&2
    exit 1
  fi
fi

LAYER_TAG="sl${SYNID_STUDENT_LAYERS//,/_}-tl${SYNID_TEACHER_LAYERS//,/_}"
RUN_TAG="e${EPOCHS}-bs${BATCH_SIZE}-lr${LR}-G${GRAD_ACC}-N${GPUS_PER_NODE}-NN${NNODES}-kd${KD_RATIO}-${SYNID_KD_LOSS}-tau${SYNID_CONTRASTIVE_TAU}-a${SYNID_ALPHA}-b${SYNID_BETA}-${SYNID_LAYER_CONFIG}-${LAYER_TAG}-pool${SYNID_POOLING}-keywords-lambda${SYNID_SYNTAX_LAMBDA}-lora-${PEFT_LORA_R}-${PEFT_LORA_ALPHA}-${PEFT_LORA_DROPOUT}"
SAVE_PATH="${SAVE_PATH:-${BASE_PATH}/results/${RUN_NAME}_spider_${KD_TYPE}_${RUN_TAG}}"

OPTS=(
  --base-path "${BASE_PATH}"
  --model-path "${MODEL_PATH}"
  --teacher-model-path "${TEACHER_MODEL_PATH}"
  --ckpt-name "${CKPT_NAME}"
  --teacher-ckpt-name "${TEACHER_CKPT_NAME}"
  --model-type "${MODEL_TYPE}"
  --teacher-model-type "${TEACHER_MODEL_TYPE}"
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
  --kd-ratio "${KD_RATIO}"
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
  --type "${KD_TYPE}"
  --synid-alpha "${SYNID_ALPHA}"
  --synid-beta "${SYNID_BETA}"
  --synid-kd-loss "${SYNID_KD_LOSS}"
  --synid-pool-tau "${SYNID_POOL_TAU}"
  --synid-contrastive-tau "${SYNID_CONTRASTIVE_TAU}"
  --synid-syntax-lambda "${SYNID_SYNTAX_LAMBDA}"
  --synid-pooling "${SYNID_POOLING}"
  --synid-use-syntax-weights "${SYNID_USE_SYNTAX_WEIGHTS}"
  --synid-use-con1 "${SYNID_USE_CON1}"
  --synid-use-con2 "${SYNID_USE_CON2}"
  --synid-use-privileged-teacher-input "${SYNID_USE_PRIVILEGED_TEACHER_INPUT}"
  --synid-student-layers "${SYNID_STUDENT_LAYERS}"
  --synid-teacher-layers "${SYNID_TEACHER_LAYERS}"
  --do-sample
  --top-k 0
  --top-p 0.95
  --temperature 0.5
  --peft lora
  --peft-lora-r "${PEFT_LORA_R}"
  --peft-lora-alpha "${PEFT_LORA_ALPHA}"
  --peft-lora-dropout "${PEFT_LORA_DROPOUT}"
)

if [[ -n "${TEACHER_PEFT_PATH}" ]]; then
  OPTS+=(--teacher-peft-path "${TEACHER_PEFT_PATH}")
fi

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}"

mkdir -p "${SAVE_PATH}"
echo "Run: ${RUN_NAME}"
echo "Student: ${MODEL_PATH}"
echo "Teacher: ${TEACHER_MODEL_PATH}"
echo "Teacher LoRA: ${TEACHER_PEFT_PATH:-none}"
echo "Data: ${DATA_DIR}"
echo "Save: ${SAVE_PATH}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Training: epochs=${EPOCHS} batch=${BATCH_SIZE} grad_acc=${GRAD_ACC} lr=${LR}"
echo "SynID: kd=${SYNID_KD_LOSS} kd_ratio=${KD_RATIO} alpha=${SYNID_ALPHA} beta=${SYNID_BETA} pool_tau=${SYNID_POOL_TAU} contrastive_tau=${SYNID_CONTRASTIVE_TAU}"
echo "Layers: student=${SYNID_STUDENT_LAYERS} teacher=${SYNID_TEACHER_LAYERS}"

CODE_BASE=HF torchrun "${DISTRIBUTED_ARGS[@]}" "${BASE_PATH}/finetuning/synid_sql_finetune.py" "${OPTS[@]}" "$@"
