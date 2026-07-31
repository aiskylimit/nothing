#! /usr/bin/env bash

set -euo pipefail

export KD_TYPE="${KD_TYPE:-sfkl}"
export SAVE_METHOD_TAG="${SAVE_METHOD_TAG:-fdd_sfkl}"
export TRAINER="${TRAINER:-finetuning/fdd_finetune.py}"
export ADD_FDD_ARGS=true
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACC="${GRAD_ACC:-16}"

source "$(dirname "${BASH_SOURCE[0]}")/../common_train.inc" "$@"
