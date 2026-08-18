#!/usr/bin/env bash
# One-time bootstrap of the EC2 instance (Ubuntu, 8xH200, driver pre-installed).
#
# Installs miniconda, creates the `pvsd` conda env, installs this package and
# flash-attn. Idempotent: after the first successful run it exits in a second, so
# commands.sh can call it before every job.
#
#   bash scripts/remote/setup_env.sh          # install if needed
#   FORCE_SETUP=1 bash scripts/remote/setup_env.sh   # redo the pip install step
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=paths.sh
source "${SCRIPT_DIR}/paths.sh"

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-pvsd}"
STAMP="${PVSD_OUT_ROOT}/.env_ready_${ENV_NAME}"

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    echo "=== installing miniconda into ${CONDA_ROOT} ==="
    installer="${PVSD_OUT_ROOT}/miniconda.sh"
    wget -q -O "${installer}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "${installer}" -b -p "${CONDA_ROOT}"
    rm -f "${installer}"
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "=== creating conda env ${ENV_NAME} (this takes a while: vllm + torch) ==="
    conda env create -f "${REPO_ROOT}/environment.yml" -n "${ENV_NAME}"
fi

conda activate "${ENV_NAME}"

if [[ ! -f "${STAMP}" || "${FORCE_SETUP:-0}" == "1" ]]; then
    echo "=== installing pvsd + flash-attn ==="
    pip install -e "${REPO_ROOT}"
    # flash-attn must build against the installed torch, hence --no-build-isolation.
    pip install flash-attn==2.8.3 --no-build-isolation
    pip install hf_transfer
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STAMP}"
fi

echo "=== environment ready ==="
python -c "import torch, transformers, trl, vllm, peft, deepspeed; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), \
'| gpus', torch.cuda.device_count()); \
print('transformers', transformers.__version__, '| trl', trl.__version__, '| vllm', vllm.__version__)"
python -c "import pvsd, pvsd.math.train, pvsd.math.pvsd_trainer; print('pvsd package imports OK')"
