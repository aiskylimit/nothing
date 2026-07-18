#! /bin/bash

set -euo pipefail

CKPT="${CKPT:-meta-llama/Llama-3.2-1B-Instruct}"
CKPT_NAME="${CKPT_NAME:-llama3.2-1b-instruct}"
USE_LORA="${USE_LORA:-0}"
RUN_GPUS="${RUN_GPUS:-0}"

source "$(dirname "${BASH_SOURCE[0]}")/common_sft.inc" "$@"
