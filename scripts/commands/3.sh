#!/usr/bin/env bash
# GPU 6 — s3 weaker signed-neg bridges (lambda_neg=0.1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=6
export MASTER_PORT=29543
export EXP_SUFFIX=s3

export KD_WEIGHT=1.0
export SEGD_K_EIGEN_MIN=16
export SEGD_TAU_INTRA=1.0
export SEGD_TAU_LOCAL=1.0
export SEGD_LAMBDA_NEG=0.1
export SEGD_DEPTH_RATIO=0.8
export SEGD_INTRA_TOPK=16

bash scripts/commands/run_one.sh
