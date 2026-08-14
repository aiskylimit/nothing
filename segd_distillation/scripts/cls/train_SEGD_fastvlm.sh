#!/bin/bash
# SEGD FastVLM classification train — hyperparams overridable via env.
#
# Example:
#   CUDA_VISIBLE_DEVICES=4 EXP_SUFFIX=baseline KD_WEIGHT=1.0 \
#     bash scripts/cls/train_SEGD_fastvlm.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

LORA_R="${LORA_R:-32}"
LORA_A="${LORA_A:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PERCENT_DATA="${PERCENT_DATA:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

KD_WEIGHT="${KD_WEIGHT:-1.0}"
SEGD_DEPTH_RATIO="${SEGD_DEPTH_RATIO:-0.8}"
SEGD_ATTN_WINDOW="${SEGD_ATTN_WINDOW:-0}"
SEGD_INTRA_TOPK="${SEGD_INTRA_TOPK:-16}"
SEGD_TAU_INTRA="${SEGD_TAU_INTRA:-1.0}"
SEGD_TAU_LOCAL="${SEGD_TAU_LOCAL:-1.0}"
SEGD_LAMBDA_NEG="${SEGD_LAMBDA_NEG:-0.3}"
SEGD_K_NEG="${SEGD_K_NEG:-8}"
SEGD_BRIDGE_TEMPERATURE="${SEGD_BRIDGE_TEMPERATURE:-1.0}"
SEGD_K_EIGEN="${SEGD_K_EIGEN:-0}"
SEGD_K_EIGEN_MIN="${SEGD_K_EIGEN_MIN:-16}"

EXP_SUFFIX="${EXP_SUFFIX:-starbridge}"
EXP_NAME="${EXP_NAME:-SEGD_FastVLM_cls_r${LORA_R}_bs${BATCH_SIZE}_${EXP_SUFFIX}}"
USE_FULLSET="${USE_FULLSET:-true}"

echo "========================================================="
echo "SEGD Training"
echo "  EXP_NAME=$EXP_NAME"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  KD_WEIGHT=$KD_WEIGHT  k_min=$SEGD_K_EIGEN_MIN"
echo "  tau_intra=$SEGD_TAU_INTRA tau_local=$SEGD_TAU_LOCAL lambda_neg=$SEGD_LAMBDA_NEG"
echo "========================================================="

if [ "$USE_FULLSET" = true ]; then
  SUBSETS=("ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397")
  echo "Training with FULL classification metatask."
else
  SUBSETS=("ImageNet_1K")
  echo "Training with SINGLE dataset (ImageNet_1K)."
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS_PER_NODE" --master_port="$MASTER_PORT" main.py \
  --model_name "apple/FastVLM-0.5B" \
  --teacher_model_name "raghavlite/B3_Qwen2_2B" \
  --lora True \
  --teacher_lora True \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_A" \
  --teacher_lora_r 8 \
  --teacher_pooling "mean" \
  --teacher_backbone "qwen2_vl" \
  --model_backbone "llava_qwen2" \
  --pooling "mean" \
  --dataset_name "TIGER-Lab/MMEB-train" \
  --subset_name "${SUBSETS[@]}" \
  --dataset_split "original" \
  --image_dir "vlm2vec_train/MMEB-train" \
  --percent_data "$PERCENT_DATA" \
  --output_dir "training/$EXP_NAME" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --bf16 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --save_strategy "epoch" \
  --seed 42 \
  --weight_decay 0.01 \
  --normalize True \
  --teacher_normalize True \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --kd_loss_type "segd_loss" \
  --kd_weight "$KD_WEIGHT" \
  --segd_depth_ratio "$SEGD_DEPTH_RATIO" \
  --segd_attn_window "$SEGD_ATTN_WINDOW" \
  --segd_intra_topk "$SEGD_INTRA_TOPK" \
  --segd_tau_intra "$SEGD_TAU_INTRA" \
  --segd_tau_local "$SEGD_TAU_LOCAL" \
  --segd_lambda_neg "$SEGD_LAMBDA_NEG" \
  --segd_k_neg "$SEGD_K_NEG" \
  --segd_bridge_temperature "$SEGD_BRIDGE_TEMPERATURE" \
  --segd_k_eigen "$SEGD_K_EIGEN" \
  --segd_k_eigen_min "$SEGD_K_EIGEN_MIN" \
  --segd_use_graph_reps_contrastive False \
  --teacher_patch_size 28 \
  --student_patch_size 64 \
  --image_resolution "low" \
  --report_to "none" \
  --run_name "$EXP_NAME"

echo "Training completed → training/$EXP_NAME"
