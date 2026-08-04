#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_DIR="${ENV_DIR:-.venv-eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${SCRIPT_DIR}/.local/share}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export UV_DATA_DIR="${UV_DATA_DIR:-${XDG_DATA_HOME}/uv}"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${UV_CACHE_DIR}" "${UV_DATA_DIR}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_SPEC="${PYTHON_BIN}"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_SPEC="$(command -v python3.10)"
elif command -v python3.9 >/dev/null 2>&1; then
  PYTHON_SPEC="$(command -v python3.9)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_SPEC="$(command -v python3)"
else
  PYTHON_SPEC="${PYTHON_VERSION}"
fi

echo "Creating eval virtual environment at ${ENV_DIR} ..."
if command -v uv >/dev/null 2>&1; then
  if ! uv venv "${ENV_DIR}" --python "${PYTHON_SPEC}"; then
    echo "uv venv failed, falling back to python3 -m venv ..." >&2
    python3 -m venv "${ENV_DIR}"
  fi
else
  echo "uv not found, using python3 -m venv ..." >&2
  python3 -m venv "${ENV_DIR}"
fi

if [[ -f "${ENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_DIR}/bin/activate"
else
  echo "Could not find ${ENV_DIR}/bin/activate" >&2
  exit 1
fi

python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install --upgrade pip

echo "Installing PyTorch GPU wheels from ${TORCH_INDEX} ..."
python -m pip install --index-url "${TORCH_INDEX}" torch torchvision

echo "Installing eval dependencies from ${PYPI_INDEX} ..."
python -m pip install --index-url "${PYPI_INDEX}" \
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
