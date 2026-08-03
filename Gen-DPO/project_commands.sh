#!/usr/bin/env bash
set -euo pipefail

#create train env
source create_train_env_uv.sh

uv pip install huggingface_hub

#download model
bash download_model.sh

#download dataset
hf download pvdhihihi/ultra-feedback \
    --repo-type dataset \
    --local-dir datasets/ultra-feedback

#train sft
bash train_sft.sh

deactivate

#create eval env
source create_eval_env_uv.sh

MODEL_DIR="$(ls -td output/sft_Llama-3.1-Tulu-3-8B-SFT_ultra-feedback_* 2>/dev/null | head -n 1 || true)"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "Could not find trained model directory under output/." >&2
  exit 1
fi

#eval sft model
MODEL_NAME="$(basename "${MODEL_DIR}")"
bash all.sh "${MODEL_NAME}"

deactivate
source .venv-tis-dpo/bin/activate

#train dpo
bash train_dpo.sh 

deactivate
source .venv-eval/bin/activate

MODEL_DIR="$(ls -td output/dpo_Llama-3.1-Tulu-3-8B-SFT_ultra-feedback_* 2>/dev/null | head -n 1 || true)"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "Could not find trained DPO model directory under output/." >&2
  exit 1
fi

#eval dpo model
MODEL_NAME="$(basename "${MODEL_DIR}")"
bash all.sh "${MODEL_NAME}"

deactivate
source .venv-tis-dpo/bin/activate

# train gen-dpo
bash train_gendpo.sh

deactivate
source .venv-eval/bin/activate

MODEL_DIR="$(ls -td output/gendpo_Llama-3.1-Tulu-3-8B-SFT_ultra-feedback_* 2>/dev/null | head -n 1 || true)"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "Could not find trained Gen-DPO model directory under output/." >&2
  exit 1
fi

# eval gen-dpo model
MODEL_NAME="$(basename "${MODEL_DIR}")"
bash all.sh "${MODEL_NAME}"
