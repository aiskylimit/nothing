#! /usr/bin/env bash

set -euo pipefail

export KD_TYPE="${KD_TYPE:-csd}"

source "$(dirname "${BASH_SOURCE[0]}")/../common_train.inc" "$@"
