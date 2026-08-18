#!/usr/bin/env bash
set -e

# Run this file after:
#   cd /media/volume/ES_volumne/dat/nothing/PAD

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"


# =========================
# 1. Optional system setup
# =========================
# Uncomment these lines if this is a fresh machine and you have sudo access.
#
# sudo apt-get update
# sudo apt-get upgrade -y


# =========================
# 2. Create Python envs and install requirements
# =========================
# PAD training is pinned to transformers==4.50.1. vLLM 0.8.x needs a newer
# transformers, so data generation and training use separate envs.

PAD_ENV="${PAD_ENV:-/media/volume/ES_volumne/dat/pad-env}"
PAD_VLLM_ENV="${PAD_VLLM_ENV:-/media/volume/ES_volumne/dat/pad-vllm-env}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/media/volume/ES_volumne/dat/.uv-cache}"

mkdir -p "$UV_CACHE_DIR"

if [[ ! -x "$PAD_ENV/bin/python" ]]; then
  uv venv "$PAD_ENV" --python 3.10
  uv pip install --python "$PAD_ENV/bin/python" -r pyproject.toml
fi

if [[ ! -x "$PAD_VLLM_ENV/bin/python" ]]; then
  uv venv "$PAD_VLLM_ENV" --python 3.10
fi

if ! "$PAD_VLLM_ENV/bin/python" - <<'PY' >/dev/null 2>&1
import transformers
import setuptools, wheel
import vllm, datasets, accelerate, numpy, tqdm, jinja2, Levenshtein

major = int(transformers.__version__.split(".", 1)[0])
if major >= 5:
    raise SystemExit(1)
PY
then
  uv pip install --python "$PAD_VLLM_ENV/bin/python" \
    "setuptools>=75" \
    "wheel>=0.44" \
    "vllm>=0.8,<0.9" \
    "transformers>=4.51.1,<5" \
    "datasets>=3.3,<4" \
    "accelerate>=1.5,<2" \
    "numpy==1.26.4" \
    "tqdm" \
    "jinja2" \
    "python-Levenshtein"
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"


# =========================
# 3. Optional Hugging Face login
# =========================
# Uncomment if you use private or gated models.
#
# source "$PAD_ENV/bin/activate"
# huggingface-cli login
# deactivate


# =========================
# 4. Models, dataset, and output paths
# =========================
# STUDENT_MODEL and TEACHER_MODEL can be Hugging Face IDs or absolute local paths.

export STUDENT_MODEL="${STUDENT_MODEL:-pvdhihihi/qwen-1.7b-sft}"
export TEACHER_MODEL="${TEACHER_MODEL:-pvdhihihi/qwen-8b-dpo}"

STUDENT_ID="${STUDENT_ID:-${STUDENT_MODEL##*/}}"
export TEACHER_ID="${TEACHER_ID:-${TEACHER_MODEL##*/}}"
export STUDENT_DIR="${STUDENT_DIR:-data/generated/ultrafeedback/$STUDENT_ID}"
export BASE_DATASET="${BASE_DATASET:-data/ultrafeedback-split}"
export DATASET_REPO="${DATASET_REPO:-pvdhihihi/ultra-feedback}"
export DATASET_CONFIG="${DATASET_CONFIG:-}"
export DATASET_SPLIT="${DATASET_SPLIT:-train}"
export SAMPLING_SEEDS="${SAMPLING_SEEDS:-0 1 2 3 4}"
export SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-4096}"
export NUM_RESPONSES="${NUM_RESPONSES:-4}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
export TRAIN_CONFIG_TEMPLATE="${TRAIN_CONFIG_TEMPLATE:-training_configs/gemma-2-2b-it-pd.yaml}"
export TRAIN_CONFIG_RUNTIME="${TRAIN_CONFIG_RUNTIME:-training_configs/generated-$STUDENT_ID-pd.yaml}"


# =========================
# 5. Optional dataset setup
# =========================
# The sampling step expects a local Hugging Face dataset with a prompt column at:
#   $BASE_DATASET
#
# If your dataset is already prepared somewhere else, set BASE_DATASET before running:
#   BASE_DATASET=/absolute/path/to/ultrafeedback-split bash project_commands.sh
#
# Otherwise, this block downloads DATASET_REPO and saves a normalized local copy.

if [[ ! -d "$BASE_DATASET" ]]; then
  "$PAD_VLLM_ENV/bin/python" - <<'PY'
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset

repo = os.environ["DATASET_REPO"]
config = os.environ.get("DATASET_CONFIG") or None
requested_split = os.environ["DATASET_SPLIT"]
output_dir = Path(os.environ["BASE_DATASET"])

if config:
    raw = load_dataset(repo, config)
else:
    raw = load_dataset(repo)

if not isinstance(raw, DatasetDict):
    raw = DatasetDict({requested_split: raw})

if requested_split not in raw:
    first_split = next(iter(raw.keys()))
    raw = DatasetDict({requested_split: raw[first_split]})

def first_user_message(messages):
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if content:
                return str(content)
    for message in messages:
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
    return None

def get_prompt(example):
    prompt = example.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return {"prompt": prompt.strip()}
    if isinstance(prompt, list):
        message_prompt = first_user_message(prompt)
        if message_prompt:
            return {"prompt": message_prompt.strip()}

    for key in ("messages", "chosen", "rejected"):
        message_prompt = first_user_message(example.get(key))
        if message_prompt:
            return {"prompt": message_prompt.strip()}

    for key in ("instruction", "question", "input"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return {"prompt": value.strip()}

    raise ValueError(
        "Cannot find a usable prompt field. "
        f"Available columns: {list(example.keys())}"
    )

processed = {}
for split_name, dataset in raw.items():
    mapped = dataset.map(
        get_prompt,
        remove_columns=dataset.column_names,
        desc=f"Normalizing {split_name}",
    )
    mapped = mapped.filter(lambda row: bool(row["prompt"]))
    processed[split_name] = mapped

normalized = DatasetDict(processed)
output_dir.parent.mkdir(parents=True, exist_ok=True)
normalized.save_to_disk(str(output_dir))
print(f"Saved local prompt dataset to {output_dir}")
print(normalized)
PY
fi

export RUN_DATASET="$BASE_DATASET"
if [[ "$SAMPLE_LIMIT" != "0" && "$SAMPLE_LIMIT" != "" ]]; then
  export RUN_DATASET="data/ultrafeedback-split-smoke-$SAMPLE_LIMIT"
  "$PAD_VLLM_ENV/bin/python" - <<'PY'
import os
from pathlib import Path

from datasets import DatasetDict, load_from_disk

base_dataset = Path(os.environ["BASE_DATASET"])
run_dataset = Path(os.environ["RUN_DATASET"])
sample_limit = int(os.environ["SAMPLE_LIMIT"])
split_name = os.environ["DATASET_SPLIT"]

if run_dataset.exists():
    raise SystemExit(0)

raw = load_from_disk(str(base_dataset))
if not isinstance(raw, DatasetDict):
    raw = DatasetDict({split_name: raw})

processed = {}
for name, dataset in raw.items():
    limit = min(sample_limit, len(dataset))
    processed[name] = dataset.select(range(limit))

run_dataset.parent.mkdir(parents=True, exist_ok=True)
DatasetDict(processed).save_to_disk(str(run_dataset))
print(f"Saved smoke-test dataset to {run_dataset}")
PY
fi


# =========================
# 6. Student sampling
# =========================
# Generates:
#   $STUDENT_DIR/output_<seed>.json

source "$PAD_VLLM_ENV/bin/activate"

export CUDA_VISIBLE_DEVICES="${SAMPLING_CUDA_VISIBLE_DEVICES:-0}"

for seed in $SAMPLING_SEEDS; do
  python data_gen/gen/sampling.py \
    --model_name "$STUDENT_MODEL" \
    --dataset "$RUN_DATASET" \
    --dataset_split train \
    --local \
    --max_tokens "$SAMPLING_MAX_TOKENS" \
    --temperature 1.0 \
    --top_p 0.95 \
    --seed "$seed" \
    --output_dir "$STUDENT_DIR"
done

deactivate


# =========================
# 7. Teacher scoring and PAD dataset generation
# =========================
# Generates:
#   $STUDENT_DIR/pkd-dataset-teacher-$TEACHER_ID-n4

source "$PAD_VLLM_ENV/bin/activate"

export CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-0}"

N="$NUM_RESPONSES"
TEMPERATURE=1

python data_gen/gen/agg.py \
  --generation_file_dir "$STUDENT_DIR" \
  -n "$N"

if (( N < 2 )); then
  echo "NUM_RESPONSES=$N only tests sampling/aggregation."
  echo "Teacher scoring and PAD training need at least 2 responses per prompt."
  echo "For a full run, use: SAMPLING_SEEDS='0 1 2 3 4' SAMPLING_MAX_TOKENS=4096 NUM_RESPONSES=4 bash $0"
  deactivate
  exit 0
fi

python data_gen/gen/prob_sl.py \
  --model_name "$TEACHER_MODEL" \
  --temperature "$TEMPERATURE" \
  --input_file "$STUDENT_DIR/agg_outputs_n$N.json" \
  --output_file "$STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.sl.json"

python data_gen/gen/prob.py \
  --model_name "$TEACHER_MODEL" \
  --temperature "$TEMPERATURE" \
  --num_options "$N" \
  --input_file "$STUDENT_DIR/agg_outputs_n$N.json" \
  --output_file "$STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.json"

python data_gen/gen/generate_dataset.py \
  --prob_file "$STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.json" \
  --prob_sl_file "$STUDENT_DIR/agg_outputs_n$N.prob.$TEACHER_ID.sl.json" \
  --output_dir "$STUDENT_DIR/pkd-dataset-teacher-$TEACHER_ID-n$N"

deactivate


# =========================
# 8. Train PAD
# =========================
# The runtime YAML is generated from TRAIN_CONFIG_TEMPLATE so model_name_or_path
# and dataset_mixer always match the values above.

source "$PAD_ENV/bin/activate"

export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
export ACCELERATE_LOG_LEVEL=info
export PAD_DATASET_DIR="$STUDENT_DIR/pkd-dataset-teacher-$TEACHER_ID-n$N"
export PAD_OUTPUT_DIR="${PAD_OUTPUT_DIR:-outputs/$STUDENT_ID-pd}"
export PAD_RUN_NAME="${PAD_RUN_NAME:-$STUDENT_ID-pd}"

python - <<'PY'
import os
from pathlib import Path

import yaml

template = Path(os.environ["TRAIN_CONFIG_TEMPLATE"])
runtime = Path(os.environ["TRAIN_CONFIG_RUNTIME"])
dataset_dir = os.environ["PAD_DATASET_DIR"]

with template.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["model_name_or_path"] = os.environ["STUDENT_MODEL"]
config["dataset_mixer"] = {dataset_dir: 1.0}
config["dataset_splits"] = ["train", "test"]
config["local_dataset"] = True
config["output_dir"] = os.environ["PAD_OUTPUT_DIR"]
config["run_name"] = os.environ["PAD_RUN_NAME"]

runtime.parent.mkdir(parents=True, exist_ok=True)
with runtime.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)

print(f"Wrote runtime training config: {runtime}")
PY

accelerate launch \
  --config_file accelerate_configs/deepspeed-zero2-2gpus-p25601.yaml \
  scripts/run_pad.py "$TRAIN_CONFIG_RUNTIME" \
  --model_name_or_path="$STUDENT_MODEL"

deactivate
