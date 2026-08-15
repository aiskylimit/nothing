#!/usr/bin/env bash
# GPU 0 — s1 baseline (λ_sim=λ_spectral=1, N=4 → 3 checkpoints)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29541
export EXP_SUFFIX=s1

export SEGD_LAMBDA_SIM=1.0
export SEGD_LAMBDA_SPECTRAL=1.0
export SEGD_TAU_GRAPH=1.0
export SEGD_NUM_ALIGN_LAYERS=4
export SEGD_K_EIGEN=0
export SEGD_K_EIGEN_MIN=16

bash scripts/commands/run_one.sh
