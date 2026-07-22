#! /usr/bin/env bash

set -euo pipefail

DISTILL_TYPE="${DISTILL_TYPE:-rkl}"
BASELINE_KIND="${BASELINE_KIND:-rkl}"

source "$(dirname "${BASH_SOURCE[0]}")/common_baseline.inc" "$@"
