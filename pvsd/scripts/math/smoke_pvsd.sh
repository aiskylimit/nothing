#!/usr/bin/env bash
# Plumbing check: does the PVSD training loop actually compose on a real model?
#
# 1 GPU, 3 optimizer steps, short rollouts, PIE cut down to a handful of passes.
# This is a plumbing check, not an experiment: it verifies that trl + DeepSpeed +
# vLLM + LoRA + the PVSD hooks run together, that the topology is detected
# correctly, and that the steered teacher differs from the student.
#
#   bash scripts/math/smoke_pvsd.sh
#
# Runtime: a few minutes. Delete ~/outputs/checkpoints/pvsd/smoke afterwards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== PVSD smoke run (plumbing only) ==="
NUM_PROCESSES="${NUM_PROCESSES:-1}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}" \
RUN_CONFIG="${RUN_CONFIG:-smoke}" \
OUTPUT_DIR="${OUTPUT_DIR:-${PVSD_CKPT_ROOT:-${HOME}/outputs/checkpoints}/pvsd/smoke}" \
    bash "${SCRIPT_DIR}/train_pvsd_qwen3_4b.sh" \
        --max_steps 3 \
        --save_steps 1000 \
        --logging_steps 1 \
        --per_device_train_batch_size 3 \
        --max_completion_length 256 \
        --max_length 2048 \
        --pvsd_views full_solution \
        --pvsd_num_corrupt 2 \
        --pvsd_pie_num_examples 1 \
        --pvsd_pie_layers 8:12 \
        --pvsd_pie_head_chunk 16 \
        --vllm_gpu_memory_utilization 0.3 \
        "$@"

cat <<'CHECKS'

=== What the log above must show ===
  Model topology: ModelTopology(num_layers=36, num_heads=32, head_dim=128, resid_dim=2560)
  Injection layer: 9 / 36
  [PVSD] PIE step 0 view=full_solution: top-10 heads [...]
  metrics containing pvsd/full_solution/cos_raw_corrupt, pvsd/full_solution/transfer_ratio,
                    pvsd/steer_advantage, pvsd/fused_norm
  loss: finite and non-zero

Three symptoms that mean the teacher is NOT being steered - stop and investigate:
  * head_dim is not 128            -> topology detection is wrong for this model
  * pvsd/steer_advantage == 0      -> the injected vector has no effect
  * loss == 0                      -> the steered teacher equals the student

If all good:  rm -rf ~/outputs/checkpoints/pvsd/smoke  and move on to scripts/math/probe_pvsd.sh
CHECKS
