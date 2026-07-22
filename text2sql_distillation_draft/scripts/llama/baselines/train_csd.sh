#! /usr/bin/env bash

set -euo pipefail

DISTILL_TYPE="${DISTILL_TYPE:-csd}"
BASELINE_KIND="${BASELINE_KIND:-csd}"

source "$(dirname "${BASH_SOURCE[0]}")/common_baseline.inc" "$@"
