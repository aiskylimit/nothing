#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="0"

export LM_EVAL_LOGLEVEL=DEBUG
export VLLM_LOGLEVEL=INFO

MODEL_NAME="${MODEL_NAME:-}"
if [[ -z "${MODEL_NAME}" ]]; then
  echo "Usage: MODEL_NAME=<model path or id> bash mmlu.sh" >&2
  exit 1
fi

TASKS="mmlu"                # Replace with your desired tasks
TP_SIZE=1                            # Number of GPUs for tensor parallelism
                             # Number of model replicas
DTYPE="auto"                          # Data type (e.g., auto, float16)
GPU_UTIL=0.9                          # GPU memory utilization
BATCH_SIZE="auto:4"
MAX_LEN=4096                     

# Construct model arguments
MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_UTIL},max_model_len=${MAX_LEN}"
MODEL_SLUG="${MODEL_SLUG:-$(basename "${MODEL_NAME}")}"

# Execute lm_eval
lm_eval --model vllm \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASKS}" \
        --num_fewshot=5 \
        --batch_size "${BATCH_SIZE}" \
        --output_path "output/${MODEL_SLUG}/${TASKS}" \
        --log_samples \
        2>&1 | tee /tmp/lm_eval_debug.log
