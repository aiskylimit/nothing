#!/bin/bash
# SEGD FastVLM classification train — 3-node semantic graph, multi-layer.
# Hyperparams and EXP_NAME overridable via env.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 EXP_SUFFIX=s1 \
#     SEGD_LAMBDA_SIM=1.0 SEGD_LAMBDA_SPECTRAL=1.0 \
#     bash scripts/cls/train_SEGD_fastvlm.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

LORA_R="${LORA_R:-64}"
LORA_A="${LORA_A:-64}"
TEACHER_LORA_R="${TEACHER_LORA_R:-64}"
# Graph is 6B × 6B per checkpoint (not native-token dense). B=16 is cheap; raise if memory allows.
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PERCENT_DATA="${PERCENT_DATA:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
PROJECTOR_LR="${PROJECTOR_LR:-5e-4}"

SEGD_LAMBDA_SIM="${SEGD_LAMBDA_SIM:-1.0}"
SEGD_LAMBDA_SPECTRAL="${SEGD_LAMBDA_SPECTRAL:-1.0}"
SEGD_TAU_GRAPH="${SEGD_TAU_GRAPH:-1.0}"
SEGD_NUM_ALIGN_LAYERS="${SEGD_NUM_ALIGN_LAYERS:-4}"
SEGD_K_EIGEN="${SEGD_K_EIGEN:-0}"
SEGD_K_EIGEN_MIN="${SEGD_K_EIGEN_MIN:-16}"

EXP_SUFFIX="${EXP_SUFFIX:-run}"
EXP_NAME="${EXP_NAME:-$EXP_SUFFIX}"
USE_FULLSET="${USE_FULLSET:-true}"

echo "========================================================="
echo "SEGD Training (3-node multi-layer)"
echo "  EXP_NAME=$EXP_NAME"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  λ_sim=$SEGD_LAMBDA_SIM  λ_spectral=$SEGD_LAMBDA_SPECTRAL"
echo "  tau_graph=$SEGD_TAU_GRAPH  num_align_layers=$SEGD_NUM_ALIGN_LAYERS"
echo "  k_min=$SEGD_K_EIGEN_MIN  k_cap=$SEGD_K_EIGEN"
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
  --teacher_lora_r "$TEACHER_LORA_R" \
  --teacher_backbone "qwen2_vl" \
  --model_backbone "llava_qwen2" \
  --pooling "eos" \
  --teacher_pooling "eos" \
  --dataset_name "TIGER-Lab/MMEB-train" \
  --subset_name "${SUBSETS[@]}" \
  --dataset_split "original" \
  --image_dir "vlm2vec_train/MMEB-train" \
  --percent_data "$PERCENT_DATA" \
  --output_dir "training/$EXP_NAME" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --projector_lr "$PROJECTOR_LR" \
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
  --segd_lambda_sim "$SEGD_LAMBDA_SIM" \
  --segd_lambda_spectral "$SEGD_LAMBDA_SPECTRAL" \
  --segd_tau_graph "$SEGD_TAU_GRAPH" \
  --segd_num_align_layers "$SEGD_NUM_ALIGN_LAYERS" \
  --segd_k_eigen "$SEGD_K_EIGEN" \
  --segd_k_eigen_min "$SEGD_K_EIGEN_MIN" \
  --image_resolution "low" \
  --report_to "none" \
  --run_name "$EXP_NAME"

echo "Training completed → training/$EXP_NAME"
