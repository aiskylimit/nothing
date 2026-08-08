#!/usr/bin/env bash
uv sync
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=4,5,6,7 python gen.py