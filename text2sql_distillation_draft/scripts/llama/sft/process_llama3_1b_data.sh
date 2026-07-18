#! /bin/bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"

source "$(dirname "${BASH_SOURCE[0]}")/process_llama_data.sh" "$@"
