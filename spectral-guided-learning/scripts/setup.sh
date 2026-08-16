#!/usr/bin/env bash
# Install the full stack (training + eval/vLLM) into a single venv on a fresh GPU box.
# Uses `uv` for a Python 3.12 venv (this box's Python 3.12 install is uv-managed; plain
# `python3.12 -m venv` fails with "externally-managed-environment").
set -euo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_PATH}"

# Check `nvidia-smi`'s "CUDA Version" (top-right) first -- that's the driver's ceiling, not a
# free choice. cu124 default is for the H200 target box; override for other hardware, e.g.
# CUDA_TAG=cu118 ./scripts/setup.sh
CUDA_TAG="${CUDA_TAG:-cu124}"
VENV_DIR=.venv
INSTALL_FLASH_ATTN=false
VLLM_VERSION=0.8.3   # newest vllm whose pinned torch (2.6.0) still has wheels for the CUDA tags
                      # above, and whose transformers floor isn't broken by transformers v5

command -v uv >/dev/null || { echo "ERROR: uv not found (needed for a Python 3.12 venv here)" >&2; exit 1; }
[[ -d "${VENV_DIR}" ]] || uv venv --python 3.12 "${VENV_DIR}"
VENV_PY="${BASE_PATH}/${VENV_DIR}/bin/python"

# torch/torchvision/torchaudio version strings carry no CUDA marker, so installing vllm alone
# can silently pull a different CUDA build for torchvision/torchaudio than the pinned torch --
# force all three from the same CUDA_TAG index together, --reinstall so a mismatched prior
# install (same version string, wrong CUDA build) actually gets replaced instead of "already
# satisfied".
uv pip install --python "${VENV_PY}" --reinstall \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
uv pip install --python "${VENV_PY}" "vllm==${VLLM_VERSION}"
# vllm 0.8.3's LoRA cache (vllm/utils.py LRUCache.touch) calls self._LRUCache__update(key),
# relying on a private method name-mangled from cachetools.LRUCache's own internals. cachetools
# 6.0 removed it (renamed/inlined) -- pulling in latest cachetools breaks LoRA eval at request
# time with "AttributeError: 'LoRALRUCache' object has no attribute '_LRUCache__update'"
# (confirmed empirically). Pin to a version that still has it.
uv pip install --python "${VENV_PY}" "cachetools==5.5.2"
# requirements.txt pins transformers>=4.56,<5 for Qwen3 support -- higher than vllm 0.8.3's own
# floor (>=4.51,<5), so this also satisfies vllm's constraint. Re-verify eval after bumping
# transformers further: vllm 0.8.3 was only tested against its own, lower floor.
uv pip install --python "${VENV_PY}" -r requirements.txt
# datasets/pandas pull the latest numpy (2.4+), but vllm's numba dep (speculative decoding,
# imported at engine startup even when unused) hard-requires numpy<2.2 -- repin after the fact.
uv pip install --python "${VENV_PY}" "numpy<2.2"
[[ "${INSTALL_FLASH_ATTN}" == true ]] && uv pip install --python "${VENV_PY}" flash-attn --no-build-isolation

"${VENV_PY}" - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible to torch"
from vllm import LLM
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda} | vllm import OK")
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  "${VENV_PY}" -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
  echo "HF login OK"
else
  echo "WARN: HF_TOKEN not set (needed for gated datasets, e.g. GPQA in evaluate.py)"
fi
echo "setup done"
