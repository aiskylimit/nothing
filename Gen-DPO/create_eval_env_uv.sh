#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-.venv-eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is not installed or not on PATH." >&2
  exit 1
}

echo "Creating eval virtual environment at ${ENV_DIR} ..."
uv venv "${ENV_DIR}" --python "${PYTHON_VERSION}"

if [[ -f "${ENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_DIR}/bin/activate"
else
  echo "Could not find ${ENV_DIR}/bin/activate" >&2
  exit 1
fi

python -m pip install --upgrade pip

echo "Installing PyTorch GPU wheels from ${TORCH_INDEX} ..."
uv pip install --index-url "${TORCH_INDEX}" torch torchvision

echo "Installing eval dependencies from ${PYPI_INDEX} ..."
uv pip install --index-url "${PYPI_INDEX}" \
  vllm \
  lm-eval \
  datasets \
  transformers \
  accelerate \
  tqdm \
  protobuf \
  sentencepiece \
  bs4 \
  beautifulsoup4

echo
echo "Done."
echo "Activate later with: source ${ENV_DIR}/bin/activate"
