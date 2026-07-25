#! /usr/bin/env bash

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
TRAIN_TYPE="${TRAIN_TYPE:-adaptive-srkl}"
SYNID_KD_LOSS="${SYNID_KD_LOSS:-srkl}"
OVERHEAD_METHOD_NAME="${OVERHEAD_METHOD_NAME:-distillm_dcr}"
SAVE_TAG="${SAVE_TAG:-qwen_ablation_4_distillm_dcr}"

export BATCH_SIZE GRAD_ACC TRAIN_TYPE SYNID_KD_LOSS OVERHEAD_METHOD_NAME SAVE_TAG

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_dcr_train.inc"
