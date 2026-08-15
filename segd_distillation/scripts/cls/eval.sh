#!/bin/bash
# Classification evaluation on MMEB-eval.
# LoRA / batch / exp name follow env (same defaults as train_SEGD_fastvlm.sh).
#
#   bash scripts/cls/eval.sh
#   bash scripts/cls/eval.sh [MODEL_PATH] [OUTPUT_DIR] [SUBSET]
#   EXP_NAME=s1 bash scripts/cls/eval.sh
# SUBSET: space-separated names, or "all" for full MMEB-eval list
set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
LORA_R="${LORA_R:-64}"
LORA_A="${LORA_A:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EXP_NAME="${EXP_NAME:-${EXP_SUFFIX:-run}}"

MODEL_PATH="${1:-training/${EXP_NAME}/checkpoint-final}"
OUTPUT_DIR="${2:-eval_outputs/${EXP_NAME}}"
SUBSET_ARG="${3:-ImageNet-1K}"
USE_FULLSET=false

if [ "$SUBSET_ARG" = "all" ]; then
    SUBSETS=("ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211")
    echo "Evaluating with FULL dataset set."
elif [ "$USE_FULLSET" = true ]; then
    SUBSETS=("ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" "Place365" "ImageNet-A" "ImageNet-R" "ObjectNet" "Country211")
    echo "Evaluating with FULL dataset set."
else
    # shellcheck disable=SC2206
    SUBSETS=($SUBSET_ARG)
    echo "Evaluating subset(s): ${SUBSETS[*]}"
fi

echo "========================================================="
echo "Starting Evaluation"
echo "Model:  ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Subset: ${SUBSETS[*]}"
echo "LoRA:   r=${LORA_R} alpha=${LORA_A}"
echo "========================================================="

echo "Detected ${NUM_GPUS_PER_NODE} GPU(s)"

EVAL_ARGS=(
    --model_name "${MODEL_PATH}"
    --encode_output_path "${OUTPUT_DIR}"
    --dataset_name "TIGER-Lab/MMEB-eval"
    --subset_name "${SUBSETS[@]}"
    --dataset_split "test"
    --per_device_eval_batch_size "${BATCH_SIZE}"
    --image_dir "eval_images/"
    --image_resolution "low"
    --pooling "mean"
    --model_backbone "llava_qwen2"
    --normalize True
    --bf16
    --tgt_prefix_mod
    --lora True
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_A}"
)

if [ "${NUM_GPUS_PER_NODE}" -gt 1 ]; then
    echo "Using multi-GPU mode with accelerate"
    accelerate launch --multi_gpu --num_processes="${NUM_GPUS_PER_NODE}" eval_mmeb.py "${EVAL_ARGS[@]}"
else
    echo "Using single GPU mode"
    python3 eval_mmeb.py "${EVAL_ARGS[@]}"
fi

echo "========================================================="
echo "Evaluation Completed"
echo "Results saved in ${OUTPUT_DIR}"
echo "========================================================="
