#! /bin/bash

set -euo pipefail

CKPT="${CKPT:-meta-llama/Llama-3.1-8B-Instruct}"
CKPT_NAME="${CKPT_NAME:-llama3.1-8b-instruct}"
USE_LORA="${USE_LORA:-1}"
RUN_GPUS="${RUN_GPUS:-0,1}"

source "$(dirname "${BASH_SOURCE[0]}")/common_sft.inc" "$@"
