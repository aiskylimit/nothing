#! /usr/bin/env bash

set -euo pipefail

export KD_TYPE="${KD_TYPE:-adaptive-amid}"
export ADD_AMID_ARGS=true

source "$(dirname "${BASH_SOURCE[0]}")/../common_train.inc" "$@"
