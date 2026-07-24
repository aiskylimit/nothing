#!/usr/bin/env bash

set -euo pipefail

MODEL_TAG="llama3.2-1b-instruct"
TEACHER_TAG="llama3.1-8b-instruct"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-https://huggingface.co/Dream-AI-HUST/llama_spider/tree/main/llama/sft_sft_llama3_8b_lora_spider_lm_e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/1090}"
TEMPLATE="${TEMPLATE:-llama3}"
LORA_TARGET="${LORA_TARGET:-q_proj,v_proj}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_train_kid_spider.sh"
