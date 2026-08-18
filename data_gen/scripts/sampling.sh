#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

HOME_PATH=$(echo ~)
source "$HOME_PATH/.bashrc"
micromamba activate pad

STUDENT_MODEL=${STUDENT_MODEL:-gemma-2-2b-it}
STUDENT_DIR=${STUDENT_DIR:-data/generated/ultrafeedback/gemma-2b-it}
for seed in 0 1 2 3 4; do
    python data_gen/gen/sampling.py --model_name "$STUDENT_MODEL" \
        --dataset data/ultrafeedback-split \
        --dataset_split train \
        --local \
        --max_tokens 4096 \
        --temperature 1.0 \
        --top_p 0.95 \
        --seed $seed \
        --output_dir "$STUDENT_DIR"
done
