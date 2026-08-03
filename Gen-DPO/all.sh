#!/bin/bash
set -e

MODEL_NAME="${1:-$MODEL_NAME}"
if [[ -z "$MODEL_NAME" ]]; then
  echo "Usage: bash run_all.sh <MODEL_NAME>"
  exit 1
fi

export MODEL_NAME

bash arc.sh
bash mmlu.sh
bash qa.sh
bash wino.sh
bash hellaswag.sh
bash gsm8k.sh
