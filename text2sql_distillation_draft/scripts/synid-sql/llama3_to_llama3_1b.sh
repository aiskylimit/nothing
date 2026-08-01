#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-synid_sql_llama3_8b_to_llama3_1b}"
export MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
export CKPT_NAME="${CKPT_NAME:-llama3.2-1b-instruct}"
export MODEL_TYPE="${MODEL_TYPE:-llama}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export TEACHER_CKPT_NAME="${TEACHER_CKPT_NAME:-llama3.1-8b-instruct}"
export TEACHER_MODEL_TYPE="${TEACHER_MODEL_TYPE:-llama}"
export TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-https://huggingface.co/Dream-AI-HUST/llama_spider/tree/main/llama/sft_sft_llama3_8b_lora_spider_lm_e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/1090}"
export DATA_DIR="${DATA_DIR:-processed_data/spider_data/synid_privileged_lora_218/llama}"

# SynID-SQL method settings.
export SYNID_KD_LOSS="${SYNID_KD_LOSS:-csd}"
export KD_RATIO="${KD_RATIO:-0.7}"
export SYNID_ALPHA="${SYNID_ALPHA:-0.3}"
export SYNID_BETA="${SYNID_BETA:-0.3}"
export SYNID_POOL_TAU="${SYNID_POOL_TAU:-5.0}"
export SYNID_CONTRASTIVE_TAU="${SYNID_CONTRASTIVE_TAU:-0.05}"
export SYNID_SYNTAX_LAMBDA="${SYNID_SYNTAX_LAMBDA:-2.0}"
export SYNID_USE_SYNTAX_WEIGHTS="${SYNID_USE_SYNTAX_WEIGHTS:-true}"
export SYNID_STUDENT_LAYERS="${SYNID_STUDENT_LAYERS:-15}"
export SYNID_TEACHER_LAYERS="${SYNID_TEACHER_LAYERS:-31}"
export SYNID_LAYER_CONFIG="${SYNID_LAYER_CONFIG:-last_layer}"

exec bash "${SCRIPT_DIR}/../common/train_synid_sql.sh" "$@"
