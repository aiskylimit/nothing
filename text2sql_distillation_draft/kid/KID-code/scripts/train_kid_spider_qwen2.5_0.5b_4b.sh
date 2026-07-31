#!/usr/bin/env bash

set -euo pipefail

MODEL_TAG="qwen2.5-0.5b"
TEACHER_TAG="qwen3-4b"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-hf://Dream-AI-HUST/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090}"
TEMPLATE="${TEMPLATE:-qwen2.5}"
LORA_TARGET="${LORA_TARGET:-q_proj,v_proj}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-1479}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-133}"
RUN_GPUS="${RUN_GPUS:-0,1}"
GRAD_ACC="${GRAD_ACC:-2}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_train_kid_spider.sh"
