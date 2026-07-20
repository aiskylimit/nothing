#! /bin/bash

set -euo pipefail

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
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
SCRIPT_GROUP="$(basename "$(dirname "${BASH_SOURCE[0]}")")"

DATA_DIR="${DATA_DIR:-processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen}"

CKPT_NAME="${CKPT_NAME:-qwen3-0.6B}"
CKPT="${CKPT:-Qwen/Qwen3-0.6B}"
TEACHER_CKPT_NAME="${TEACHER_CKPT_NAME:-qwen3-4B}"
TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen3-4B-Instruct-2507}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-hf://distillation-sql/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090}"

BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-0.0001}"
GRAD_ACC="${GRAD_ACC:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
KD_RATIO="${KD_RATIO:-0.7}"
KD_TYPE="${KD_TYPE:-synid}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
T_MAX_LENGTH="${T_MAX_LENGTH:-2048}"
T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-1800}"

PEFT_LORA_R="${PEFT_LORA_R:-16}"
PEFT_LORA_ALPHA="${PEFT_LORA_ALPHA:-64}"
PEFT_LORA_DROPOUT="${PEFT_LORA_DROPOUT:-0.1}"

SYNID_ALPHA="${SYNID_ALPHA:-0.1}"
SYNID_BETA="${SYNID_BETA:-0.1}"
SYNID_KD_LOSS="${SYNID_KD_LOSS:-csd}"
SYNID_POOL_TAU="${SYNID_POOL_TAU:-5}"
SYNID_CONTRASTIVE_TAU="${SYNID_CONTRASTIVE_TAU:-0.05}"
SYNID_SYNTAX_LAMBDA="${SYNID_SYNTAX_LAMBDA:-1.0}"
SYNID_POOLING="${SYNID_POOLING:-sc}"
SYNID_USE_SYNTAX_WEIGHTS="${SYNID_USE_SYNTAX_WEIGHTS:-false}"
SYNID_USE_CON1="${SYNID_USE_CON1:-true}"
SYNID_USE_CON2="${SYNID_USE_CON2:-true}"

SYNID_STUDENT_LAYERS="${SYNID_STUDENT_LAYERS:-27}"
SYNID_TEACHER_LAYERS="${SYNID_TEACHER_LAYERS:-35}"
SYNID_LAYER_CONFIG="${SYNID_LAYER_CONFIG:-k1_last_s27_t35}"

LAYER_TAG="sl${SYNID_STUDENT_LAYERS//,/_}-tl${SYNID_TEACHER_LAYERS//,/_}"
RUN_TAG="e${EPOCHS}-bs${BATCH_SIZE}-lr${LR}-G${GRAD_ACC}-N${GPUS_PER_NODE}-NN${NNODES}-kd${KD_RATIO}-${SYNID_KD_LOSS}-tau${SYNID_CONTRASTIVE_TAU}-a${SYNID_ALPHA}-b${SYNID_BETA}-${SYNID_LAYER_CONFIG}-${LAYER_TAG}-pool${SYNID_POOLING}-lora-${PEFT_LORA_R}-${PEFT_LORA_ALPHA}-${PEFT_LORA_DROPOUT}"
SAVE_TAG="${SAVE_TAG:-${SCRIPT_GROUP}_${SCRIPT_NAME}_spider_${KD_TYPE}_${RUN_TAG}}"
SAVE_PATH="${SAVE_PATH:-${BASE_PATH}/results/qwen3/${SAVE_TAG}}"
SEED="${SEED:-42}"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
# if [[ -n "${TEACHER_PEFT_PATH}" ]]; then
#   OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
# fi
OPTS+=" --model-type qwen"
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
OPTS+=" --type ${KD_TYPE}"
OPTS+=" --synid-alpha ${SYNID_ALPHA}"
OPTS+=" --synid-beta ${SYNID_BETA}"
OPTS+=" --synid-kd-loss ${SYNID_KD_LOSS}"
OPTS+=" --synid-pool-tau ${SYNID_POOL_TAU}"
OPTS+=" --synid-contrastive-tau ${SYNID_CONTRASTIVE_TAU}"
OPTS+=" --synid-syntax-lambda ${SYNID_SYNTAX_LAMBDA}"
OPTS+=" --synid-pooling ${SYNID_POOLING}"
OPTS+=" --synid-use-syntax-weights ${SYNID_USE_SYNTAX_WEIGHTS}"
OPTS+=" --synid-use-con1 ${SYNID_USE_CON1}"
OPTS+=" --synid-use-con2 ${SYNID_USE_CON2}"
OPTS+=" --synid-student-layers ${SYNID_STUDENT_LAYERS}"
OPTS+=" --synid-teacher-layers ${SYNID_TEACHER_LAYERS}"
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r ${PEFT_LORA_R}"
OPTS+=" --peft-lora-alpha ${PEFT_LORA_ALPHA}"
OPTS+=" --peft-lora-dropout ${PEFT_LORA_DROPOUT}"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${BASE_PATH}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetuning/synid_sql_finetune.py ${OPTS} $*"

echo "${CMD}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "SynID layer config: ${SYNID_LAYER_CONFIG}"
echo "  student layers: ${SYNID_STUDENT_LAYERS}"
echo "  teacher layers: ${SYNID_TEACHER_LAYERS}"
echo "Length config:"
echo "  max length: ${MAX_LENGTH}"
echo "  max prompt length: ${MAX_PROMPT_LENGTH}"
echo "  teacher max length: ${T_MAX_LENGTH}"
echo "  teacher max prompt length: ${T_MAX_PROMPT_LENGTH}"
mkdir -p "${SAVE_PATH}"
CODE_BASE=HF ${CMD}
