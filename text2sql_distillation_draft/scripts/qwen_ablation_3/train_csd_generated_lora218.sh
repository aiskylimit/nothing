#! /usr/bin/env bash

set -euo pipefail

DATA_DIR="${DATA_DIR:-processed_data/benchmarks/spider_data/generated_lora_218_train_only/qwen}"
SAVE_TAG="${SAVE_TAG:-qwen_ablation_3_csd_generated_lora218_train_only_spider}"

export DATA_DIR
export SAVE_TAG

echo "Qwen ablation 3 CSD on generated LoRA-218 train-only data"
echo "  data dir: ${DATA_DIR}"
echo "  save tag: ${SAVE_TAG}"

bash scripts/kd_2/csd/train_0.6b_4b.sh "$@"
