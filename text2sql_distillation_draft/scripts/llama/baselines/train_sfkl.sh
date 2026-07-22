#! /usr/bin/env bash

set -euo pipefail

DISTILL_TYPE="${DISTILL_TYPE:-sfkl}"
BASELINE_KIND="${BASELINE_KIND:-sfkl}"

source "$(dirname "${BASH_SOURCE[0]}")/common_baseline.inc" "$@"
