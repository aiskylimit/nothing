#!/usr/bin/env bash
# GPU 7 — s4 lower eigengap k floor (k_eigen_min=8)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=7
export MASTER_PORT=29544
export EXP_SUFFIX=s4

export KD_WEIGHT=1.0
export SEGD_K_EIGEN_MIN=8
export SEGD_TAU_INTRA=1.0
export SEGD_TAU_LOCAL=1.0
export SEGD_LAMBDA_NEG=0.3
export SEGD_DEPTH_RATIO=0.8
export SEGD_INTRA_TOPK=16

bash scripts/commands/run_one.sh
