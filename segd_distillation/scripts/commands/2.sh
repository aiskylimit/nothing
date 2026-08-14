#!/usr/bin/env bash
# GPU 5 — s2 stronger KD (kd_weight=2)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=5
export MASTER_PORT=29542
export EXP_SUFFIX=s2

export KD_WEIGHT=2.0
export SEGD_K_EIGEN_MIN=16
export SEGD_TAU_INTRA=1.0
export SEGD_TAU_LOCAL=1.0
export SEGD_LAMBDA_NEG=0.3
export SEGD_DEPTH_RATIO=0.8
export SEGD_INTRA_TOPK=16

bash scripts/commands/run_one.sh
