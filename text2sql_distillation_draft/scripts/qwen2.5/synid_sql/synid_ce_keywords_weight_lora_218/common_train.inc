#! /usr/bin/env bash

set -euo pipefail

ENTRY_SCRIPT="${BASH_SOURCE[1]}"
SCRIPT_DIR="$(cd "$(dirname "${ENTRY_SCRIPT}")" && pwd)"
SCRIPT_GROUP="$(basename "${SCRIPT_DIR}")"
SCRIPT_NAME="$(basename "${ENTRY_SCRIPT}" .sh)"
BASE_TRAIN_SCRIPT="${SCRIPT_DIR}/train_0.5b_4b_csd_multilayer.sh"

GRID_ID="${GRID_ID:-${SCRIPT_NAME#train_}}"
GRID_ID="${GRID_ID^^}"
GRID_NUM="${GRID_ID#G}"
GRID_NUM="$((10#${GRID_NUM}))"

KD_RATIOS=(0.7)
LAYER_CONFIGS=(lm_head)
DATA_TAG="lora218-qwen-shared"
DATA_DIR="${DATA_DIR:-processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen}"
GRID_TOTAL="$((${#LAYER_CONFIGS[@]} * ${#KD_RATIOS[@]}))"

if (( GRID_NUM < 1 || GRID_NUM > GRID_TOTAL )); then
  echo "Unknown GRID_ID=${GRID_ID}. Expected G01." >&2
  exit 2
fi

GRID_INDEX="$((GRID_NUM - 1))"
KD_INDEX="$((GRID_INDEX % ${#KD_RATIOS[@]}))"
LAYER_INDEX="$((GRID_INDEX / ${#KD_RATIOS[@]}))"
LAYER_CONFIG="${LAYER_CONFIGS[LAYER_INDEX]}"

if [[ "${LAYER_CONFIG}" == "lm_head" ]]; then
  SYNID_K_LAYERS="1"
  SYNID_STUDENT_LAYERS="-1"
  SYNID_TEACHER_LAYERS="-1"
  SYNID_LAYER_CONFIG="k1_lm_head_s-1_t-1"
else
  echo "Unsupported LAYER_CONFIG=${LAYER_CONFIG}" >&2
  exit 2
fi

SYNID_ALPHA="${SYNID_ALPHA:-0.3}"
SYNID_BETA="${SYNID_BETA:-0.3}"
SYNID_CONTRASTIVE_TAU="${SYNID_CONTRASTIVE_TAU:-0.05}"
SYNID_USE_SYNTAX_WEIGHTS="${SYNID_USE_SYNTAX_WEIGHTS:-true}"
SYNID_SYNTAX_LAMBDA="${SYNID_SYNTAX_LAMBDA:-2.0}"
KD_RATIO="${KD_RATIOS[KD_INDEX]}"

export DATA_DIR
export DATA_TAG
export SYNID_STUDENT_LAYERS
export SYNID_TEACHER_LAYERS
export SYNID_ALPHA
export SYNID_BETA
export SYNID_CONTRASTIVE_TAU
export SYNID_LAYER_CONFIG
export SYNID_USE_SYNTAX_WEIGHTS
export SYNID_SYNTAX_LAMBDA
export SYNID_KD_LOSS="csd"
export KD_RATIO

RUN_TAG="data${DATA_TAG}-e${EPOCHS:-5}-bs${BATCH_SIZE:-4}-lr${LR:-0.0001}-G${GRAD_ACC:-1}-grid${GRID_ID}-k${SYNID_K_LAYERS}-kd${KD_RATIO}-${SYNID_KD_LOSS}-tau${SYNID_CONTRASTIVE_TAU}-a${SYNID_ALPHA}-b${SYNID_BETA}-${SYNID_LAYER_CONFIG}-pool${SYNID_POOLING:-sc}-keywords-lambda${SYNID_SYNTAX_LAMBDA}-lora-${PEFT_LORA_R:-16}-${PEFT_LORA_ALPHA:-64}-${PEFT_LORA_DROPOUT:-0.1}"
export SAVE_TAG="${SAVE_TAG:-qwen2.5_${SCRIPT_GROUP}_${SCRIPT_NAME}_spider_${KD_TYPE:-synid}_${RUN_TAG}}"

echo "Qwen2.5 <- Qwen3 shared-tokenizer SynID grid: ${GRID_ID}"
echo "  data tag: ${DATA_TAG}"
echo "  data dir: ${DATA_DIR}"
echo "  k final layers: ${SYNID_K_LAYERS}"
echo "  student layers: ${SYNID_STUDENT_LAYERS}"
echo "  teacher layers: ${SYNID_TEACHER_LAYERS}"
echo "  alpha: ${SYNID_ALPHA}"
echo "  kd ratio: ${KD_RATIO}"
echo "  beta: ${SYNID_BETA}"
echo "  contrastive tau: ${SYNID_CONTRASTIVE_TAU}"
echo "  use syntax weights: ${SYNID_USE_SYNTAX_WEIGHTS}"
echo "  syntax lambda: ${SYNID_SYNTAX_LAMBDA}"
echo "  save tag: ${SAVE_TAG}"

bash "${BASE_TRAIN_SCRIPT}" "$@"
