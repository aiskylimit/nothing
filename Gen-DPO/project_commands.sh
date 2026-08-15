#!/usr/bin/env bash
set -euo pipefail

uv sync
source .venv/bin/activate

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

#download model
bash download_model.sh

#download dataset
hf download pvdhihihi/ultra-feedback \
    --repo-type dataset \
    --local-dir datasets/ultra-feedback

#train sft
bash train_sft.sh

#train dpo
bash train_dpo.sh 

# train gen-dpo
bash train_gendpo.sh