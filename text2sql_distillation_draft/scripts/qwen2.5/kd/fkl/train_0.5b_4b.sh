#! /usr/bin/env bash

set -euo pipefail

export KD_TYPE="${KD_TYPE:-fkl}"

source "$(dirname "${BASH_SOURCE[0]}")/../common_train.inc" "$@"
