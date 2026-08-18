#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../remote/paths.sh
source "${REPO_ROOT}/scripts/remote/paths.sh"

RUN_CONFIG="${RUN_CONFIG:-qwen3_4b_pvsd}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
EXTRA_ARGS=("$@")

cmd=(
    accelerate launch
    --config_file configs/accelerate.yaml
    --num_processes "${NUM_PROCESSES:-4}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --main_process_port "${MAIN_PROCESS_PORT:-12949}"
    -m pvsd.math.train
    --model_name_or_path Qwen/Qwen3-4B
    --training_dataset openthought
    --learning_rate 5e-6
    --max_grad_norm 0.1
    --per_device_train_batch_size 4
    --gradient_checkpointing
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --output_dir "${OUTPUT_DIR:-${PVSD_CKPT_ROOT}/pvsd/qwen3_4b}"
    --run_config "${RUN_CONFIG}"
    --max_steps 500
    --max_completion_length 4096
    --save_steps 10
    --save_only_model true
    --logging_steps 2
    --attn_implementation flash_attention_2
    --torch_dtype bfloat16
    --max_length 16384
    --use_vllm
    --vllm_mode colocate
    --vllm_gpu_memory_utilization 0.4
    --vllm_tensor_parallel_size 1
    --use_peft
    --lora_r 64
    --lora_alpha 128
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --temperature 0.6
    --top_p 0.95
    --top_k 0
    --beta 1.0
    --jsd_token_clip 0
    --multi_view_mode single
    --pvsd_enable
    --pvsd_views full_solution,partial_solution,answer_only
    --pvsd_num_corrupt 2
    --pvsd_layer_fraction quarter
    --pvsd_alpha 1.0
    --pvsd_steer_scope completion
    --pvsd_extract_micro_batch 8
    --pvsd_top_k_heads 10
    --pvsd_pie_every 100
    --pvsd_pie_num_examples 2
    --pvsd_pie_head_chunk 8
    --pvsd_pie_layers all
)

if ((${#EXTRA_ARGS[@]} > 0)); then
    cmd+=("${EXTRA_ARGS[@]}")
fi

echo "=== Running ${RUN_CONFIG} ==="
(
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" \
        "${cmd[@]}"
)
