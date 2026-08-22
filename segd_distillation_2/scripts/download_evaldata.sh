#!/usr/bin/env bash
# Download full MMEB-eval images → eval_images/
# Usage (from repo root): bash scripts/download_evaldata.sh
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -q hf_transfer 2>/dev/null || true
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download TIGER-Lab/MMEB-eval images.zip --repo-type dataset --local-dir .
mkdir -p eval_images
unzip -o images.zip -d eval_images/
rm -f images.zip
echo "done → eval_images/"
