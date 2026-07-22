#! /usr/bin/env bash

set -euo pipefail

ENTRY_SCRIPT="${BASH_SOURCE[1]}"
SCRIPT_DIR="$(cd "$(dirname "${ENTRY_SCRIPT}")" && pwd)"
SCRIPT_GROUP="$(basename "${SCRIPT_DIR}")"
SCRIPT_NAME="$(basename "${ENTRY_SCRIPT}" .sh)"
BASE_TRAIN_SCRIPT="${SCRIPT_DIR}/../qwen_updated_2/synid_ce_keywords_weight_lora_218/train_0.6b_4b_csd_multilayer.sh"

# Fixed to qwen_updated_2/synid_ce_keywords_weight_lora_218 G01.
DATA_TAG="lora218"
DATA_DIR="processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen"
KD_RATIO="0.7"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
SYNID_K_LAYERS="1"
SYNID_STUDENT_LAYERS="27"
SYNID_TEACHER_LAYERS="35"
SYNID_LAYER_CONFIG="k1_last_s27_t35"
SYNID_ALPHA="0.3"
SYNID_BETA="0.3"
SYNID_CONTRASTIVE_TAU="${SYNID_CONTRASTIVE_TAU:-0.05}"
SYNID_USE_CON1="true"
SYNID_USE_CON2="true"
SYNID_USE_SYNTAX_WEIGHTS="false"
SYNID_SYNTAX_LAMBDA="${SYNID_SYNTAX_LAMBDA:-1.0}"
SYNID_KD_LOSS="csd"

export DATA_DIR
export DATA_TAG
export KD_RATIO
export BATCH_SIZE
export GRAD_ACC
export SYNID_KD_LOSS
export SYNID_STUDENT_LAYERS
export SYNID_TEACHER_LAYERS
export SYNID_ALPHA
export SYNID_BETA
export SYNID_CONTRASTIVE_TAU
export SYNID_LAYER_CONFIG
export SYNID_USE_CON1
export SYNID_USE_CON2
export SYNID_USE_SYNTAX_WEIGHTS
export SYNID_SYNTAX_LAMBDA

ABLATION_ID="no_syntax_weight"
RUN_TAG="data${DATA_TAG}-e${EPOCHS:-5}-bs${BATCH_SIZE}-lr${LR:-0.0001}-G${GRAD_ACC}-g01-k${SYNID_K_LAYERS}-kd${KD_RATIO}-${SYNID_KD_LOSS}-tau${SYNID_CONTRASTIVE_TAU}-a${SYNID_ALPHA}-b${SYNID_BETA}-${ABLATION_ID}-${SYNID_LAYER_CONFIG}-pool${SYNID_POOLING:-sc}-nokeywords-lambda${SYNID_SYNTAX_LAMBDA}-lora-${PEFT_LORA_R:-16}-${PEFT_LORA_ALPHA:-64}-${PEFT_LORA_DROPOUT:-0.1}"
export SAVE_TAG="${SCRIPT_GROUP}_${SCRIPT_NAME}_spider_${KD_TYPE:-synid}_${RUN_TAG}"
unset SAVE_PATH

echo "Qwen ablation 2: ${ABLATION_ID}"
echo "  base setting: qwen_updated_2/synid_ce_keywords_weight_lora_218 G01"
echo "  data tag: ${DATA_TAG}"
echo "  data dir: ${DATA_DIR}"
echo "  k final layers: ${SYNID_K_LAYERS}"
echo "  student layers: ${SYNID_STUDENT_LAYERS}"
echo "  teacher layers: ${SYNID_TEACHER_LAYERS}"
echo "  alpha: ${SYNID_ALPHA}"
echo "  beta: ${SYNID_BETA}"
echo "  kd ratio: ${KD_RATIO}"
echo "  batch size: ${BATCH_SIZE}"
echo "  grad acc: ${GRAD_ACC}"
echo "  kd loss: ${SYNID_KD_LOSS}"
echo "  contrastive tau: ${SYNID_CONTRASTIVE_TAU}"
echo "  use con1: ${SYNID_USE_CON1}"
echo "  use con2: ${SYNID_USE_CON2}"
echo "  use syntax weights: ${SYNID_USE_SYNTAX_WEIGHTS}"
echo "  syntax lambda: ${SYNID_SYNTAX_LAMBDA}"
echo "  save tag: ${SAVE_TAG}"

bash "${BASE_TRAIN_SCRIPT}" "$@"
