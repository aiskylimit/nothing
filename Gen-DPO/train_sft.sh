#!/usr/bin/env bash
set -euo pipefail

exec python -u train.py \
  model=llama8b \
  model.name_or_path="Llama-3.1-Tulu-3-8B-SFT" \
  datasets='[ultra-feedback]' \
  loss=sft \
  gradient_accumulation_steps=4 \
  batch_size=8 \
  eval_batch_size=8 \
  lr=2e-4 \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  base_data_dir=datasets/ \
  reverse_dataset=false
