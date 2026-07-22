#! /usr/bin/env bash

set -euo pipefail

DISTILL_TYPE="${DISTILL_TYPE:-srkl}"
BASELINE_KIND="${BASELINE_KIND:-srkl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-16}"

source "$(dirname "${BASH_SOURCE[0]}")/common_baseline.inc" "$@"
