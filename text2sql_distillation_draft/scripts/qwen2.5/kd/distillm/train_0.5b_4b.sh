#! /usr/bin/env bash

set -euo pipefail

export KD_TYPE="${KD_TYPE:-adaptive-srkl}"
export SAVE_METHOD_TAG="${SAVE_METHOD_TAG:-distillm_adaptive-srkl}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACC="${GRAD_ACC:-4}"

source "$(dirname "${BASH_SOURCE[0]}")/../common_train.inc" "$@"
