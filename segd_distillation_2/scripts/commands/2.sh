#!/usr/bin/env bash
# GPU 1 — s2 stronger KD (λ_sim=λ_spectral=2)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=29542
export EXP_SUFFIX=s2

export SEGD_LAMBDA_SIM=2.0
export SEGD_LAMBDA_SPECTRAL=2.0
export SEGD_TAU_GRAPH=1.0
export SEGD_NUM_ALIGN_LAYERS=4
export SEGD_K_EIGEN=0
export SEGD_K_EIGEN_MIN=16

bash scripts/commands/run_one.sh
