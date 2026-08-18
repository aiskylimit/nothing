#!/usr/bin/env bash
# Shared paths for every PVSD script.
#
# The remote runner requires that nothing large is written inside the code folder:
# checkpoints, datasets, model weights and results all live under ${PVSD_OUT_ROOT}
# (default ~/outputs), which is outside ~/pvsd/ and is what `#2 -f-` pulls from.
#
# Source it, do not execute it:  source scripts/remote/paths.sh

PVSD_OUT_ROOT="${PVSD_OUT_ROOT:-${HOME}/outputs}"
PVSD_CKPT_ROOT="${PVSD_CKPT_ROOT:-${PVSD_OUT_ROOT}/checkpoints}"
PVSD_RESULTS_ROOT="${PVSD_RESULTS_ROOT:-${PVSD_OUT_ROOT}/results}"
export PVSD_OUT_ROOT PVSD_CKPT_ROOT PVSD_RESULTS_ROOT

# HuggingFace downloads models and datasets on the instance; keep that cache out of
# the repo too (a single Qwen3-4B snapshot is larger than the 25MB repo limit).
export HF_HOME="${HF_HOME:-${PVSD_OUT_ROOT}/hf_cache}"
# Fast Hub downloads, but only when the optional package is actually installed:
# huggingface_hub errors out if the flag is set without it.
if [[ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]] && python -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${PVSD_OUT_ROOT}" "${PVSD_CKPT_ROOT}" "${PVSD_RESULTS_ROOT}" "${HF_HOME}"
