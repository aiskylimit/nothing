#!/usr/bin/env bash

export UV_PROJECT_ENVIRONMENT=vlm_distill
uv sync --locked

source vlm_distill/bin/activate

export CUDA_VISIBLE_DEVICES=4,5,6,7
uv run bash script_train/run_baseline.sh
