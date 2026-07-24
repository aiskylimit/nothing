#!/usr/bin/env bash

set -euo pipefail

MODEL_TAG="qwen3-0.6b"
TEACHER_TAG="qwen3-4b"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-hf://distillation-sql/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090}"
TEMPLATE="${TEMPLATE:-qwen3}"
LORA_TARGET="${LORA_TARGET:-q_proj,v_proj}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_train_kid_spider.sh"
