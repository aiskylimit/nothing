# SynID-SQL

This repository contains training and evaluation code for SynID-SQL, a
teacher-student distillation method for Spider-style text-to-SQL tasks.

The checked-in data is organized under `datasets/` and is split by usage:

```text
datasets/
|-- train/
|   `-- spider_data/
|       |-- train_spider.json
|       |-- train_others.json
|       |-- train_gold.sql
|       |-- tables.json
|       |-- synid_aug_v2_lora/final_merged.jsonl
|       `-- llama_synid_aug_v2_lora/final_merged.jsonl
|-- eval/
|   `-- spider_data/
|       |-- dev.json
|       |-- dev_gold.sql
|       `-- tables.json
`-- test/
    |-- spider_data/
    |   |-- test.json
    |   |-- test_gold.sql
    |   |-- tables.json
    |   `-- test_tables.json
    |-- spider_syn/test.json
    |-- spider_realistic/test.json
    `-- spider_dk/
        |-- test.json
        `-- tables.json
```

Generated SynID data lives only in `datasets/train/spider_data/`. The same
SynID train set is provided for both model families:

| Model family | Generated train source |
| --- | --- |
| Qwen | `datasets/train/spider_data/synid_aug_v2_lora/final_merged.jsonl` |
| Llama | `datasets/train/spider_data/llama_synid_aug_v2_lora/final_merged.jsonl` |

## Download Sources

Download sources and local placement under `datasets/`:

| Benchmark | Source | JSON/schema path | DB path |
| --- | --- | --- | --- |
| Spider original | [Spider 1.0](https://yale-lily.github.io/spider) | `datasets/train/spider_data/`, `datasets/eval/spider_data/`, `datasets/test/spider_data/` | `datasets/train/spider_data/database/`, `datasets/eval/spider_data/database/`, `datasets/test/spider_data/database/` |
| Spider-Syn | [ygan/Spider-Syn](https://github.com/ygan/Spider-Syn) | `datasets/test/spider_syn/test.json` | `datasets/test/spider_data/database/` |
| Spider-Realistic | [aherntech/spider-realistic](https://github.com/aherntech/spider-realistic) | `datasets/test/spider_realistic/test.json` | `datasets/test/spider_data/database/` |
| Spider-DK | [ygan/Spider-DK](https://github.com/ygan/Spider-DK) | `datasets/test/spider_dk/test.json`, `datasets/test/spider_dk/tables.json` | `datasets/test/spider_dk/database/` |

Spider-Syn and Spider-Realistic reuse the original Spider SQLite databases, so
keep those test databases in `datasets/test/spider_data/database/`. Spider-DK
uses its own database directory under `datasets/test/spider_dk/database/`.

Step 1: format Spider JSON files into prompt/response JSONL. Run this before any
`process_data.py` command, because `process_data.py` reads `train.jsonl`,
`dev.jsonl`, and `test.jsonl` from the formatted data directory.

## Environment

Use Linux or WSL for training scripts. The shell wrappers use `bash`,
`torchrun`, DeepSpeed, and CUDA/NCCL.

```bash
conda create -n synid-sql python=3.10
conda activate synid-sql
pip install -r requirements.txt
```

For the vLLM augmentation runner, install `vllm` separately in a compatible
Linux CUDA environment. The standard augmentation runner does not require
vLLM.

## Format Data

Format Spider train, eval, and test JSON files into prompt/response JSONL:

```bash
python scripts/format_spider_jsonl.py \
  --root datasets/train/spider_data \
  --splits train

python scripts/format_spider_jsonl.py \
  --root datasets/eval/spider_data \
  --splits dev

python scripts/format_spider_jsonl.py \
  --root datasets/test/spider_data \
  --splits test
```

Format the test-only robustness sets:

```bash
python scripts/format_spider_variant_jsonl.py \
  --root datasets/test \
  --split test
```

Build privileged teacher prompts for the original Spider train set:

```bash
python scripts/format_spider_synid_jsonl.py \
  --root datasets/train/spider_data \
  --output datasets/train/spider_data/format_data/teacher_train.jsonl \
  --student-train datasets/train/spider_data/format_data/train.jsonl
```

## Tokenize Base Data

`process_data.py` writes mmap `*.bin/*.idx` files to
`--processed-data-dir/<model-type>` unless the output path contains
`generated`. The resulting directory is the `DATA_DIR` for training scripts.

For `--split train`, if `--data-dir` contains `teacher_train.jsonl`,
`process_data.py` also writes `teacher_train_0.bin` and
`teacher_train_0.idx`.

Base Qwen data:

```bash
python process_data.py \
  --model-path Qwen/Qwen3-0.6B \
  --model-type qwen \
  --data-dir datasets/train/spider_data/format_data \
  --processed-data-dir processed_data/spider_data \
  --split train \
  --max-length 2048 \
  --max-prompt-length 1536 \
  --t-max-length 2048 \
  --t-max-prompt-length 1800 \
  --data-process-workers 8

python process_data.py \
  --model-path Qwen/Qwen3-0.6B \
  --model-type qwen \
  --data-dir datasets/eval/spider_data/format_data \
  --processed-data-dir processed_data/spider_data \
  --split valid \
  --max-length 2048 \
  --max-prompt-length 1536 \
  --t-max-length 2048 \
  --t-max-prompt-length 1800 \
  --data-process-workers 8

python process_data.py \
  --model-path Qwen/Qwen3-0.6B \
  --model-type qwen \
  --data-dir datasets/test/spider_data/format_data \
  --processed-data-dir processed_data/spider_data \
  --split test \
  --max-length 2048 \
  --max-prompt-length 1536 \
  --t-max-length 2048 \
  --t-max-prompt-length 1800 \
  --data-process-workers 8
```

Base Llama data uses the same input paths:

```bash
python process_data.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --model-type llama \
  --data-dir datasets/train/spider_data/format_data \
  --processed-data-dir processed_data/spider_data \
  --split train \
  --max-length 2048 \
  --max-prompt-length 1536 \
  --t-max-length 2048 \
  --t-max-prompt-length 1800 \
  --data-process-workers 8

python process_data.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --model-type llama \
  --data-dir datasets/eval/spider_data/format_data \
  --processed-data-dir processed_data/spider_data \
  --split valid \
  --max-length 2048 \
  --max-prompt-length 1536 \
  --t-max-length 2048 \
  --t-max-prompt-length 1800 \
  --data-process-workers 8

python process_data.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --model-type llama \
  --data-dir datasets/test/spider_data/format_data \
  --processed-data-dir processed_data/spider_data \
  --split test \
  --max-length 2048 \
  --max-prompt-length 1536 \
  --t-max-length 2048 \
  --t-max-prompt-length 1800 \
  --data-process-workers 8
```

Expected base output examples:

```text
processed_data/spider_data/qwen/train_0.bin
processed_data/spider_data/qwen/train_0.idx
processed_data/spider_data/qwen/teacher_train_0.bin
processed_data/spider_data/qwen/teacher_train_0.idx
processed_data/spider_data/llama/train_0.bin
processed_data/spider_data/llama/train_0.idx
```

## SynID Train Data

For the same SynID train set, build one processed source for Qwen and one for
Llama:

| Model family | Generated input | Train output |
| --- | --- | --- |
| Qwen | `datasets/train/spider_data/synid_aug_v2_lora/final_merged.jsonl` | `processed_data/spider_data/synid_privileged_lora/qwen/` |
| Llama | `datasets/train/spider_data/llama_synid_aug_v2_lora/final_merged.jsonl` | `processed_data/spider_data/synid_privileged_lora/llama/` |

Build the Qwen variant:

```bash
SYNID_QWEN_DIR=processed_data/spider_data/synid_privileged_lora/qwen
mkdir -p "${SYNID_QWEN_DIR}"
cp datasets/train/spider_data/format_data/train.jsonl "${SYNID_QWEN_DIR}/train.jsonl"
cp datasets/eval/spider_data/format_data/dev.jsonl "${SYNID_QWEN_DIR}/dev.jsonl"
cp datasets/test/spider_data/format_data/test.jsonl "${SYNID_QWEN_DIR}/test.jsonl"

python scripts/synid_augment/build_teacher_train_from_final_merged.py \
  --input datasets/train/spider_data/synid_aug_v2_lora/final_merged.jsonl \
  --output "${SYNID_QWEN_DIR}/teacher_train.jsonl" \
  --train-output "${SYNID_QWEN_DIR}/train.jsonl"
```

Tokenize the Qwen variant:

```bash
for split in train valid test; do
  python process_data.py \
    --model-path Qwen/Qwen3-0.6B \
    --model-type qwen \
    --data-dir processed_data/spider_data/synid_privileged_lora/qwen \
    --processed-data-dir processed_data/spider_data/synid_privileged_lora \
    --split "${split}" \
    --max-length 2048 \
    --max-prompt-length 1536 \
    --t-max-length 2048 \
    --t-max-prompt-length 1800 \
    --data-process-workers 8
done
```

Build the Llama variant:

```bash
SYNID_LLAMA_DIR=processed_data/spider_data/synid_privileged_lora/llama
mkdir -p "${SYNID_LLAMA_DIR}"
cp datasets/train/spider_data/format_data/train.jsonl "${SYNID_LLAMA_DIR}/train.jsonl"
cp datasets/eval/spider_data/format_data/dev.jsonl "${SYNID_LLAMA_DIR}/dev.jsonl"
cp datasets/test/spider_data/format_data/test.jsonl "${SYNID_LLAMA_DIR}/test.jsonl"

python scripts/synid_augment/build_teacher_train_from_final_merged.py \
  --input datasets/train/spider_data/llama_synid_aug_v2_lora/final_merged.jsonl \
  --output "${SYNID_LLAMA_DIR}/teacher_train.jsonl" \
  --train-output "${SYNID_LLAMA_DIR}/train.jsonl"
```

Tokenize the Llama variant:

```bash
for split in train valid test; do
  python process_data.py \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --model-type llama \
    --data-dir processed_data/spider_data/synid_privileged_lora/llama \
    --processed-data-dir processed_data/spider_data/synid_privileged_lora \
    --split "${split}" \
    --max-length 2048 \
    --max-prompt-length 1536 \
    --t-max-length 2048 \
    --t-max-prompt-length 1800 \
    --data-process-workers 8
done
```

Expected SynID output examples:

```text
processed_data/spider_data/synid_privileged_lora/qwen/train_0.bin
processed_data/spider_data/synid_privileged_lora/qwen/train_0.idx
processed_data/spider_data/synid_privileged_lora/qwen/teacher_train_0.bin
processed_data/spider_data/synid_privileged_lora/qwen/teacher_train_0.idx
processed_data/spider_data/synid_privileged_lora/llama/train_0.bin
processed_data/spider_data/synid_privileged_lora/llama/train_0.idx
```

## Training

All training wrappers inherit the same default setup:

- epochs: `5`
- batch size: `4`
- gradient accumulation: `1`
- learning rate: `1e-4`
- warmup ratio: `0.1`
- scheduler: `wrmup_cosine`
- weight decay: `1e-2`
- gradient clipping: `1.0`
- LoRA: `r=16`, `alpha=64`, `dropout=0.1`
- student max length: `2048`
- student prompt length: `1536`
- teacher max length: `2048`
- teacher prompt length: `1800`

Select GPUs with `RUN_GPUS`:

```bash
export RUN_GPUS=0,1
```

The shell wrappers require a processed data directory:

```bash
export DATA_DIR=<processed_mmap_data_dir>
export TEACHER_PEFT_PATH=<your_teacher_lora_adapter_path>
```

Teacher LoRA:

Use the processed base data directory as `DATA_DIR`: Qwen uses
`processed_data/spider_data/qwen`, and Llama uses
`processed_data/spider_data/llama`.

```bash
DATA_DIR=processed_data/spider_data/qwen \
  bash scripts/teacher_lora/qwen3.sh

DATA_DIR=processed_data/spider_data/llama \
  bash scripts/teacher_lora/llama3.sh
```

Student SFT:

Use the same processed base data directories for student SFT: Qwen uses
`processed_data/spider_data/qwen`, and Llama uses
`processed_data/spider_data/llama`.

```bash
DATA_DIR=processed_data/spider_data/qwen \
  bash scripts/student_sft/qwen3_0.6b.sh

DATA_DIR=processed_data/spider_data/llama \
  bash scripts/student_sft/llama3_1b.sh
```

SynID-SQL distillation:

After tokenizing the SynID variant, set `DATA_DIR` to that processed SynID
directory. After teacher LoRA training finishes, set `TEACHER_PEFT_PATH` to
that teacher LoRA adapter checkpoint before running the distillation script.
Use the Qwen `DATA_DIR` and Qwen teacher adapter for the Qwen distillation run;
use the Llama `DATA_DIR` and Llama teacher adapter for the Llama distillation
run.

```bash
DATA_DIR=processed_data/spider_data/synid_privileged_lora/qwen \
TEACHER_PEFT_PATH=<your_qwen_teacher_lora_adapter_path> \
  bash scripts/synid-sql/qwen3_to_qwen3_0.6b.sh

DATA_DIR=processed_data/spider_data/synid_privileged_lora/llama \
TEACHER_PEFT_PATH=<your_llama_teacher_lora_adapter_path> \
  bash scripts/synid-sql/llama3_to_llama3_1b.sh
```

## Inference and Evaluation

Inference runs five reporting seeds by default:

```text
10, 42, 50, 100, 1234
```

It evaluates on:

- `datasets/test/spider_data/test.json`, max new tokens `856`
- `datasets/test/spider_syn/test.json`, max new tokens `756`
- `datasets/test/spider_realistic/test.json`, max new tokens `755`
- `datasets/test/spider_dk/test.json`, max new tokens `663`

Set the checkpoint path before running inference:

```bash
export CHECKPOINT_PATH=results/<run_dir>/<checkpoint_or_adapter>
```

Teacher LoRA:

```bash
bash scripts/teacher_lora/infer_qwen3.sh
bash scripts/teacher_lora/eval_qwen3.sh

bash scripts/teacher_lora/infer_llama3.sh
bash scripts/teacher_lora/eval_llama3.sh
```

Student SFT:

```bash
bash scripts/student_sft/infer_qwen3_0.6b.sh
bash scripts/student_sft/eval_qwen3_0.6b.sh

bash scripts/student_sft/infer_llama3_1b.sh
bash scripts/student_sft/eval_llama3_1b.sh
```

SynID-SQL:

```bash
bash scripts/synid-sql/infer_qwen3_to_qwen3_0.6b.sh
bash scripts/synid-sql/eval_qwen3_to_qwen3_0.6b.sh

bash scripts/synid-sql/infer_llama3_to_llama3_1b.sh
bash scripts/synid-sql/eval_llama3_to_llama3_1b.sh
```

Or evaluate an existing inference run directly:

```bash
bash eval.sh synid_sql_qwen3_4b_to_qwen3_0.6b
```

Override seeds if needed:

```bash
export INFER_SEEDS=10,42,50,100,1234
```

## Generate SynID Data

Note: the checked-in `final_merged.jsonl` files are already generated. If you
want to regenerate them, first follow the download/setup steps above and make
sure the original Spider SQLite databases are available at
`datasets/train/spider_data/database/`. The vLLM generator should be run on CUDA
Linux/WSL.

Generate the Qwen SynID data:

```bash
export TEACHER_PEFT_PATH=<your_qwen_teacher_lora_adapter_path>

python scripts/synid_augment/run_spider_aug_loops_v2.py \
  --benchmark spider \
  --root datasets/train/spider_data \
  --output-root datasets/train/spider_data/synid_aug_v2_lora \
  --db-root datasets/train/spider_data/database \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --teacher-peft-path "${TEACHER_PEFT_PATH}" \
  --tensor-parallel-size 2 \
  --similarity-threshold 0.9 \
  --num-loops 5 \
  --resume

python scripts/synid_augment/overall_symthetic.py \
  --base-dir datasets/train/spider_data/synid_aug_v2_lora
```

Generate the Llama SynID data:

```bash
export TEACHER_PEFT_PATH=<your_llama_teacher_lora_adapter_path>

python scripts/synid_augment/run_spider_aug_loops_v2.py \
  --benchmark spider \
  --root datasets/train/spider_data \
  --output-root datasets/train/spider_data/llama_synid_aug_v2_lora \
  --db-root datasets/train/spider_data/database \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --teacher-peft-path "${TEACHER_PEFT_PATH}" \
  --tensor-parallel-size 2 \
  --similarity-threshold 0.9 \
  --num-loops 5 \
  --resume

python scripts/synid_augment/overall_symthetic.py \
  --base-dir datasets/train/spider_data/llama_synid_aug_v2_lora
```

After `final_merged.jsonl` is regenerated, run the commands in
`SynID Train Data` to build `teacher_train.jsonl` and tokenize the Qwen/Llama
processed data.

## Notes

- Training and vLLM augmentation are intended for CUDA Linux/WSL.
- Windows can be used for lightweight formatting and unit tests, but not for
  the full DeepSpeed/vLLM training workflow.
- Hugging Face gated models such as Llama require prior access and login.
