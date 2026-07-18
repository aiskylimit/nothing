#! /usr/bin/env bash

set -euo pipefail

ENTRY_SCRIPT="${BASH_SOURCE[1]}"
SCRIPT_DIR="$(cd "$(dirname "${ENTRY_SCRIPT}")" && pwd)"
SCRIPT_GROUP="$(basename "${SCRIPT_DIR}")"
SCRIPT_NAME="$(basename "${ENTRY_SCRIPT}" .sh)"
BASE_TRAIN_SCRIPT="${SCRIPT_DIR}/train_0.6b_4b_csd_multilayer.sh"

GRID_ID="${GRID_ID:-${SCRIPT_NAME#train_}}"
GRID_ID="${GRID_ID^^}"
GRID_NUM="${GRID_ID#G}"
GRID_NUM="$((10#${GRID_NUM}))"

ALPHAS=(0.3 0.5 0.7 1)
KD_RATIOS=(0.7 0.6 0.5 0.8 0.9)
GRID_TOTAL="$((${#ALPHAS[@]} * ${#KD_RATIOS[@]}))"

if (( GRID_NUM < 1 || GRID_NUM > GRID_TOTAL )); then
  echo "Unknown GRID_ID=${GRID_ID}. Expected G01..G20." >&2
  exit 2
fi

GRID_INDEX="$((GRID_NUM - 1))"
ALPHA_INDEX="$((GRID_INDEX / ${#KD_RATIOS[@]}))"
KD_INDEX="$((GRID_INDEX % ${#KD_RATIOS[@]}))"

SYNID_K_LAYERS="3"
SYNID_STUDENT_LAYERS="25,26,27"
SYNID_TEACHER_LAYERS="33,34,35"
SYNID_ALPHA="${ALPHAS[ALPHA_INDEX]}"
SYNID_BETA="${SYNID_BETA:-0.1}"
SYNID_CONTRASTIVE_TAU="${SYNID_CONTRASTIVE_TAU:-0.05}"
SYNID_LAYER_CONFIG="k${SYNID_K_LAYERS}_last_s25_26_27_t33_34_35"
KD_RATIO="${KD_RATIOS[KD_INDEX]}"

export SYNID_STUDENT_LAYERS
export SYNID_TEACHER_LAYERS
export SYNID_ALPHA
export SYNID_BETA
export SYNID_CONTRASTIVE_TAU
export SYNID_LAYER_CONFIG
export SYNID_KD_LOSS="csd"
export KD_RATIO

RUN_TAG="e${EPOCHS:-5}-bs${BATCH_SIZE:-8}-lr${LR:-0.0001}-G${GRAD_ACC:-1}-grid${GRID_ID}-k${SYNID_K_LAYERS}-kd${KD_RATIO}-${SYNID_KD_LOSS}-tau${SYNID_CONTRASTIVE_TAU}-a${SYNID_ALPHA}-b${SYNID_BETA}-${SYNID_LAYER_CONFIG}-pool${SYNID_POOLING:-sc}-lora-${PEFT_LORA_R:-16}-${PEFT_LORA_ALPHA:-64}-${PEFT_LORA_DROPOUT:-0.1}"
export SAVE_TAG="${SAVE_TAG:-${SCRIPT_GROUP}_${SCRIPT_NAME}_spider_${KD_TYPE:-synid}_${RUN_TAG}}"

echo "SynID multilayer grid: ${GRID_ID}"
echo "  k final layers: ${SYNID_K_LAYERS}"
echo "  student layers: ${SYNID_STUDENT_LAYERS}"
echo "  teacher layers: ${SYNID_TEACHER_LAYERS}"
echo "  alpha: ${SYNID_ALPHA}"
echo "  kd ratio: ${KD_RATIO}"
echo "  beta: ${SYNID_BETA}"
echo "  contrastive tau: ${SYNID_CONTRASTIVE_TAU}"
echo "  save tag: ${SAVE_TAG}"

bash "${BASE_TRAIN_SCRIPT}" "$@"