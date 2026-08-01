#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-synid_sql_qwen3_4b_to_qwen3_0.6b}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
export CKPT_NAME="${CKPT_NAME:-qwen3-0.6b}"
export MODEL_TYPE="${MODEL_TYPE:-qwen}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export TEACHER_CKPT_NAME="${TEACHER_CKPT_NAME:-qwen3-4b}"
export TEACHER_MODEL_TYPE="${TEACHER_MODEL_TYPE:-qwen}"
export TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-hf://Dream-AI-HUST/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/218}"
export DATA_DIR="${DATA_DIR:-processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen}"
export SYNID_STUDENT_LAYERS="${SYNID_STUDENT_LAYERS:-27}"
export SYNID_TEACHER_LAYERS="${SYNID_TEACHER_LAYERS:-35}"
export SYNID_LAYER_CONFIG="${SYNID_LAYER_CONFIG:-k1_last_s27_t35}"

exec bash "${SCRIPT_DIR}/../common/train_synid_sql.sh" "$@"
