#!/usr/bin/env bash
# Shared once: env + full MMEB train/eval images.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH=/usr/local/cuda/bin:${PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

nvidia-smi || true

echo "==> [setup] python env (3.11)"
uv python install 3.11
PY311="$(uv python find 3.11)"
if [[ -d vlm ]]; then
  # Recreate if existing venv is not 3.11
  if ! vlm/bin/python -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; then
    echo "==> [setup] existing vlm is not Python 3.11 — recreating"
    rm -rf vlm
  fi
fi
if [[ ! -d vlm ]]; then
  "$PY311" -m venv vlm
fi
# shellcheck disable=SC1091
source vlm/bin/activate
python -c 'import sys; assert sys.version_info[:2]==(3,11), sys.version'
pip install -r requirements.txt
python fix_lib.py

echo "==> [setup] full MMEB-train images"
python scripts/download_traindata.py

echo "==> [setup] full MMEB-eval images"
bash scripts/download_evaldata.sh

echo "==> [setup] done"
