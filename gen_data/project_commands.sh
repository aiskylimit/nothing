#!/usr/bin/env bash
uv sync
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 uv run python gen.py