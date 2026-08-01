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
export TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-}"
export DATA_DIR="${DATA_DIR:-}"

# SynID-SQL method settings.
export SYNID_KD_LOSS="${SYNID_KD_LOSS:-csd}"
export KD_RATIO="${KD_RATIO:-0.7}"
export SYNID_ALPHA="${SYNID_ALPHA:-0.3}"
export SYNID_BETA="${SYNID_BETA:-0.3}"
export SYNID_POOL_TAU="${SYNID_POOL_TAU:-5.0}"
export SYNID_CONTRASTIVE_TAU="${SYNID_CONTRASTIVE_TAU:-0.05}"
export SYNID_SYNTAX_LAMBDA="${SYNID_SYNTAX_LAMBDA:-2.0}"
export SYNID_USE_SYNTAX_WEIGHTS="${SYNID_USE_SYNTAX_WEIGHTS:-true}"
export SYNID_STUDENT_LAYERS="${SYNID_STUDENT_LAYERS:-27}"
export SYNID_TEACHER_LAYERS="${SYNID_TEACHER_LAYERS:-35}"
export SYNID_LAYER_CONFIG="${SYNID_LAYER_CONFIG:-last_layer}"

exec bash "${SCRIPT_DIR}/../common/train_synid_sql.sh" "$@"
