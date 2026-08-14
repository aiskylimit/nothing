#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

exec python -u train.py \
  model=llama8b \
  model.name_or_path="Llama-3.1-Tulu-3-8B-SFT" \
  datasets='[ultra-feedback]' \
  loss=dpo \
  loss.beta=0.1 \
  gradient_accumulation_steps=4 \
  batch_size=8 \
  eval_batch_size=8 \
  lr=5e-7 \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  base_data_dir=datasets/ \
  reverse_dataset=false
