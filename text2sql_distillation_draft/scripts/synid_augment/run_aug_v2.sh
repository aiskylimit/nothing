#! /usr/bin/env bash

set -euo pipefail

: "${TEACHER_PEFT_PATH:?Set TEACHER_PEFT_PATH to your custom teacher LoRA adapter path.}"

python scripts/synid_augment/run_spider_aug_loops_v2.py \
  --benchmark spider \
  --tensor-parallel-size 2 \
  --teacher-peft-path "${TEACHER_PEFT_PATH}" \
  --similarity-threshold 0.9
