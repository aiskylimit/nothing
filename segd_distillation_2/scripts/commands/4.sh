#!/usr/bin/env bash
# GPU 3 — s4 more depth checkpoints (N=5 → 4 aligned layers at 20/40/60/80%)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=29544
export EXP_SUFFIX=s4

export SEGD_LAMBDA_SIM=0.0
export SEGD_LAMBDA_SPECTRAL=0.0
export SEGD_TAU_GRAPH=1.0
export SEGD_NUM_ALIGN_LAYERS=5
export SEGD_K_EIGEN=0
export SEGD_K_EIGEN_MIN=16

bash scripts/commands/run_one.sh
