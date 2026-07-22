#! /usr/bin/env bash

set -euo pipefail

BASELINE_KIND="${BASELINE_KIND:-fdd}"
ENABLE_FDD=1
FDD_DISTILL_TYPE="${FDD_DISTILL_TYPE:-sfkl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-16}"

source "$(dirname "${BASH_SOURCE[0]}")/common_baseline.inc" "$@"
