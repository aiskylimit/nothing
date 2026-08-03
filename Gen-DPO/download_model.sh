#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="allenai/Llama-3.1-Tulu-3-8B-SFT"
LOCAL_DIR="${LOCAL_DIR:-$PWD/Llama-3.1-Tulu-3-8B-SFT}"

if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli)
elif command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf)
else
  echo "huggingface-cli/hf not found. Install huggingface_hub first." >&2
  exit 1
fi

mkdir -p "${LOCAL_DIR}"

echo "Downloading ${MODEL_ID} to ${LOCAL_DIR}"
"${HF_CLI[@]}" download \
  --resume-download \
  "${MODEL_ID}" \
  --local-dir "${LOCAL_DIR}" \
  --local-dir-use-symlinks False

echo
echo "Done."
echo "Model saved at: ${LOCAL_DIR}"
