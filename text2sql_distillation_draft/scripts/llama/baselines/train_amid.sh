#! /usr/bin/env bash

set -euo pipefail

DISTILL_TYPE="${DISTILL_TYPE:-adaptive-amid}"
BASELINE_KIND="${BASELINE_KIND:-amid}"
ENABLE_AMID=1

source "$(dirname "${BASH_SOURCE[0]}")/common_baseline.inc" "$@"
