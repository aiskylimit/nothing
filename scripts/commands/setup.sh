#!/usr/bin/env bash
# Shared once: env + full MMEB train/eval images.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

nvidia-smi || true

echo "==> [setup] python env"
uv python install 3.11 || true
if [[ ! -d vlm ]]; then
  python3 -m venv vlm
fi
# shellcheck disable=SC1091
source vlm/bin/activate
pip install -r requirements.txt
python fix_lib.py

echo "==> [setup] full MMEB-train images"
bash scripts/download_traindata/download_traindata.sh
bash scripts/download_traindata/download_traindata_2.sh

echo "==> [setup] full MMEB-eval images"
bash scripts/download_evaldata.sh

echo "==> [setup] done"
