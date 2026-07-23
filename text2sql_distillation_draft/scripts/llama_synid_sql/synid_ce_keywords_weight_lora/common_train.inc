#! /usr/bin/env bash

set -euo pipefail

ENTRY_SCRIPT="${BASH_SOURCE[1]}"
SCRIPT_DIR="$(cd "$(dirname "${ENTRY_SCRIPT}")" && pwd)"
SCRIPT_GROUP="$(basename "${SCRIPT_DIR}")"
SCRIPT_NAME="$(basename "${ENTRY_SCRIPT}" .sh)"
BASE_TRAIN_SCRIPT="${SCRIPT_DIR}/train_1b_8b_csd_synid.sh"

GRID_ID="${GRID_ID:-${SCRIPT_NAME#train_}}"
GRID_ID="${GRID_ID^^}"

if [[ "${GRID_ID}" != "G01" && "${GRID_ID}" != "G02" ]]; then
  echo "Unknown GRID_ID=${GRID_ID}. Expected G01 or G02." >&2
  exit 2
fi

SYNID_DATASET_NAME="${SYNID_DATASET_NAME:-synid_privileged_lora}"
DATA_TAG="${DATA_TAG:-lora}"
DATA_DIR="${DATA_DIR:-processed_data/benchmarks/spider_data/${SYNID_DATASET_NAME}/llama}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
KD_RATIO="${KD_RATIO:-0.7}"
SYNID_K_LAYERS="1"
SYNID_STUDENT_LAYERS="${SYNID_STUDENT_LAYERS:-15}"
SYNID_TEACHER_LAYERS="${SYNID_TEACHER_LAYERS:-31}"
SYNID_LAYER_CONFIG="${SYNID_LAYER_CONFIG:-k1_last_s15_t31}"
case "${GRID_ID}" in
  G01)
    DEFAULT_SYNID_ALPHA="0.3"
    DEFAULT_SYNID_BETA="0.3"
    ;;
  G02)
    DEFAULT_SYNID_ALPHA="0.1"
    DEFAULT_SYNID_BETA="0.1"
    ;;
esac
SYNID_ALPHA="${SYNID_ALPHA:-${DEFAULT_SYNID_ALPHA}}"
SYNID_BETA="${SYNID_BETA:-${DEFAULT_SYNID_BETA}}"
SYNID_POOL_TAU="${SYNID_POOL_TAU:-5}"
SYNID_CONTRASTIVE_TAU="${SYNID_CONTRASTIVE_TAU:-0.05}"
SYNID_USE_SYNTAX_WEIGHTS="${SYNID_USE_SYNTAX_WEIGHTS:-true}"
SYNID_SYNTAX_LAMBDA="${SYNID_SYNTAX_LAMBDA:-2.0}"
SYNID_KD_LOSS="${SYNID_KD_LOSS:-csd}"

export SYNID_DATASET_NAME
export DATA_TAG
export DATA_DIR
export BATCH_SIZE
export GRAD_ACC
export KD_RATIO
export SYNID_STUDENT_LAYERS
export SYNID_TEACHER_LAYERS
export SYNID_ALPHA
export SYNID_BETA
export SYNID_POOL_TAU
export SYNID_CONTRASTIVE_TAU
export SYNID_LAYER_CONFIG
export SYNID_USE_SYNTAX_WEIGHTS
export SYNID_SYNTAX_LAMBDA
export SYNID_KD_LOSS

RUN_TAG="data${DATA_TAG}-e${EPOCHS:-5}-bs${BATCH_SIZE}-lr${LR:-0.0001}-G${GRAD_ACC}-grid${GRID_ID}-k${SYNID_K_LAYERS}-kd${KD_RATIO}-${SYNID_KD_LOSS}-tau${SYNID_CONTRASTIVE_TAU}-a${SYNID_ALPHA}-b${SYNID_BETA}-${SYNID_LAYER_CONFIG}-pool${SYNID_POOLING:-sc}-keywords-lambda${SYNID_SYNTAX_LAMBDA}-lora-${PEFT_LORA_R:-16}-${PEFT_LORA_ALPHA:-64}-${PEFT_LORA_DROPOUT:-0.1}"
export SAVE_TAG="${SAVE_TAG:-${SCRIPT_GROUP}_${SCRIPT_NAME}_spider_${KD_TYPE:-synid}_${RUN_TAG}}"

echo "Llama SynID-SQL keywords-weight grid: ${GRID_ID}"
echo "  data tag: ${DATA_TAG}"
echo "  data dir: ${DATA_DIR}"
echo "  k final layers: ${SYNID_K_LAYERS}"
echo "  student layers: ${SYNID_STUDENT_LAYERS}"
echo "  teacher layers: ${SYNID_TEACHER_LAYERS}"
echo "  alpha: ${SYNID_ALPHA}"
echo "  beta: ${SYNID_BETA}"
echo "  pool tau: ${SYNID_POOL_TAU}"
echo "  contrastive tau: ${SYNID_CONTRASTIVE_TAU}"
echo "  kd ratio: ${KD_RATIO}"
echo "  use syntax weights: ${SYNID_USE_SYNTAX_WEIGHTS}"
echo "  syntax lambda: ${SYNID_SYNTAX_LAMBDA}"
echo "  batch size: ${BATCH_SIZE}"
echo "  grad acc: ${GRAD_ACC}"
echo "  save tag: ${SAVE_TAG}"

bash "${BASE_TRAIN_SCRIPT}" "$@"
