#!/usr/bin/env bash

uv sync
source .venv/bin/activate
uv pip install lm_eval["longbench"]

bash longbench.sh

bash qwen_4b.sh