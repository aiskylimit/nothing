#!/bin/bash

NUM_GPUS_PER_NODE=1
TRAIN_SCRIPT="train_distill_ddp.py"

EXP_NAME="${EXP_NAME:-FastVLM-0.5B_segd_eos_cls}"

torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
    --model_name apple/FastVLM-0.5B \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/$EXP_NAME" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.05 \
    --kd_loss_type "segd_loss" \
    --segd_lambda_sim 1.0 \
    --segd_lambda_spectral 1.0 \
    --segd_tau_graph 1.0 \
    --segd_num_align_layers 4 \
    --segd_k_eigen 0 \
    --segd_k_eigen_min 8 \
    --image_resolution "low" \
    --num_projectors 1 \
    --projector_lr 5e-5 \
    --report_to None
