#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${ENV_DIR:-.venv-tis-dpo}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is not installed or not on PATH." >&2
  exit 1
}

echo "Creating virtual environment at ${ENV_DIR} ..."
uv venv "${ENV_DIR}" --python "${PYTHON_VERSION}"

if [[ -f "${ENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_DIR}/bin/activate"
else
  echo "Could not find ${ENV_DIR}/bin/activate" >&2
  exit 1
fi

python -m pip install --upgrade "pip==23.0"

echo "Installing PyTorch GPU wheels from ${TORCH_INDEX} ..."
uv pip install --index-url "${TORCH_INDEX}" torch torchvision

echo "Installing project dependencies from ${PYPI_INDEX} ..."
uv pip install --index-url "${PYPI_INDEX}" \
  numpy \
  "transformers>=4.31.0" \
  "datasets>=2.12.0" \
  "hydra-core>=1.3.2" \
  "omegaconf>=2.3.0" \
  "tqdm>=4.65.0" \
  "accelerate>=0.20.3" \
  "bs4>=0.0.1" \
  "beautifulsoup4>=4.12.2" \
  "tensor-parallel>=1.3.0" \
  "sentencepiece>=0.1.99" \
  "protobuf>=4.23.3" \
  'fschat[model_worker,webui]'

echo
echo "Done."
