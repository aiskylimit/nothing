#!/usr/bin/env bash
# Train → eval → summarize for one setting (env must be set by caller).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source vlm/bin/activate
export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
: "${EXP_SUFFIX:?set EXP_SUFFIX}"
: "${MASTER_PORT:?set MASTER_PORT}"

export LORA_R="${LORA_R:-64}"
export LORA_A="${LORA_A:-64}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export EXP_NAME="${EXP_NAME:-$EXP_SUFFIX}"

CKPT="training/${EXP_NAME}/checkpoint-final"
EVAL_OUT="eval_outputs/${EXP_NAME}"
SUMMARY_TXT="results/${EXP_NAME}_eval_summary.txt"

echo "==> [${EXP_SUFFIX}] GPU=${CUDA_VISIBLE_DEVICES} train"
bash scripts/cls/train_SEGD_fastvlm.sh

echo "==> [${EXP_SUFFIX}] eval"
bash scripts/cls/eval.sh "$CKPT" "$EVAL_OUT" all

echo "==> [${EXP_SUFFIX}] summarize"
mkdir -p results
python scripts/summarize_eval.py "$EVAL_OUT" \
  -o "$SUMMARY_TXT" \
  --meta "setting=${EXP_SUFFIX}" \
  --meta "gpu=${CUDA_VISIBLE_DEVICES}" \
  --meta "exp=${EXP_NAME}" \
  --meta "ckpt=${CKPT}" \
  --meta "lambda_sim=${SEGD_LAMBDA_SIM:-}" \
  --meta "lambda_spectral=${SEGD_LAMBDA_SPECTRAL:-}" \
  --meta "tau_graph=${SEGD_TAU_GRAPH:-}" \
  --meta "num_align_layers=${SEGD_NUM_ALIGN_LAYERS:-}" \
  --meta "k_eigen_min=${SEGD_K_EIGEN_MIN:-}"

echo "==> [${EXP_SUFFIX}] done → $SUMMARY_TXT"
