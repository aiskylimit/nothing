#! /usr/bin/env bash

set -euo pipefail

DISTILL_TYPE="${DISTILL_TYPE:-fkl}"
BASELINE_KIND="${BASELINE_KIND:-fkl}"

source "$(dirname "${BASH_SOURCE[0]}")/common_baseline.inc" "$@"
