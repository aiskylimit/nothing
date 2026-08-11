#!/usr/bin/env bash
uv sync
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python gen_v2.py