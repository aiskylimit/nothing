# SynID-SQL

This repository contains the training and evaluation code for SynID-SQL, a
teacher-student distillation method for Spider-style text-to-SQL tasks.

The current release is scoped to Spider and Spider robustness variants:
Spider, Spider-Syn, Spider-Realistic, and Spider-DK.

## Repository Layout

```text
.
|-- finetuning/
|   |-- finetune.py              # SFT / LM training entry point
|   `-- synid_sql_finetune.py    # SynID-SQL distillation training entry point
|-- src/synid_sql/
|   |-- losses.py                # SynID-SQL KD and contrastive objectives
|   |-- hidden_states.py         # hidden-state/layer capture utilities
|   `-- augmentation/            # SQL reformulation and validation utilities
|-- scripts/
|   |-- teacher_lora/            # teacher LoRA training/inference/eval wrappers
|   |-- student_sft/             # student SFT training/inference/eval wrappers
|   |-- synid-sql/               # SynID-SQL training/inference/eval wrappers
|   |-- synid_augment/           # SynID SQL reformulation pipeline
|   `-- common/                  # shared train/infer/eval shell helpers
|-- prompts/single_turn/
|-- benchmarks/                  # SQLite databases and evaluator resources
`-- benchmarks_2/                # Spider-style JSON data
```

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

## Data Preparation

Expected Spider layout:

```text
benchmarks_2/
|-- spider_data/
|   |-- train_spider.json
|   |-- dev.json
|   |-- test.json
|   |-- tables.json
|   `-- test_tables.json
|-- spider_syn/test.json
|-- spider_realistic/test.json
`-- spider_dk/test.json

benchmarks/spider_data/database/
```

Format Spider JSON files into prompt/response JSONL:

```bash
python scripts/format_spider_jsonl.py \
  --root benchmarks_2/spider_data \
  --splits train dev test

python scripts/format_spider_variant_jsonl.py \
  --root benchmarks_2 \
  --split test
```

Build privileged teacher prompts for the original Spider train set:

```bash
python scripts/format_spider_synid_jsonl.py \
  --root benchmarks_2/spider_data \
  --output benchmarks_2/spider_data/format_data/teacher_train.jsonl \
  --student-train benchmarks_2/spider_data/format_data/train.jsonl
```

Tokenize each split before training. Example for Qwen:

```bash
for split in train valid test; do
  python process_data.py \
    --model-path Qwen/Qwen3-0.6B \
    --model-type qwen \
    --data-dir benchmarks_2/spider_data/format_data \
    --processed-data-dir processed_data/benchmarks/spider_data \
    --split "${split}" \
    --max-length 2048 \
    --max-prompt-length 1536 \
    --t-max-length 2048 \
    --t-max-prompt-length 1800
done
```

The `valid` split reads `dev.jsonl`. For Llama runs, set `--model-path` and
`--model-type llama`, and use the matching processed-data directory.

## SynID SQL Reformulation

The augmentation pipeline generates alternative SQL queries from a LoRA-tuned
teacher and validates them with the Spider execution evaluator.

Selection rule:

1. Generate one candidate from the teacher.
2. Discard and retry if it is not execution-equivalent to the gold SQL.
3. Accept immediately if it is execution-equivalent and has normalized SQL
   `difflib.SequenceMatcher` similarity below `0.8`.
4. Retry up to five times.
5. If no candidate is accepted but execution-correct high-similarity candidates
   exist, keep the one with the lowest similarity.
6. If no execution-correct candidate exists, fall back to the original gold SQL.

Run augmentation with the standard Transformers backend:

```bash
python scripts/synid_augment/run_aug_loops.py \
  --benchmark spider \
  --root benchmarks_2/spider_data \
  --db-root benchmarks/spider_data/database \
  --output-root benchmarks_2/spider_data/synid_aug \
  --num-loops 5 \
  --similarity-threshold 0.8
```

Or use the vLLM backend:

```bash
python scripts/synid_augment/run_spider_aug_loops_v2.py \
  --benchmark spider \
  --root benchmarks_2/spider_data \
  --db-root benchmarks/spider_data/database \
  --output-root benchmarks_2/spider_data/synid_aug_v2_lora \
  --tensor-parallel-size 2 \
  --num-loops 5 \
  --similarity-threshold 0.8
```

Merge accepted rows with recovered final rejections:

```bash
python scripts/synid_augment/overall_symthetic.py \
  --base-dir benchmarks_2/spider_data/synid_aug_v2_lora
```

Build SynID train files from the merged augmentation output:

```bash
python scripts/synid_augment/build_teacher_train_from_final_merged.py \
  --input benchmarks_2/spider_data/synid_aug_v2_lora/final_merged.jsonl \
  --output processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen/teacher_train.jsonl \
  --train-output processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen/train.jsonl
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

### Teacher LoRA

```bash
bash scripts/teacher_lora/qwen3.sh
bash scripts/teacher_lora/llama3.sh
```

### Student SFT

```bash
bash scripts/student_sft/qwen3_0.6b.sh
bash scripts/student_sft/llama3_1b.sh
bash scripts/student_sft/qwen2.5_0.5b.sh
```

### SynID-SQL Distillation

```bash
bash scripts/synid-sql/qwen3_to_qwen3_0.6b.sh
bash scripts/synid-sql/llama3_to_llama3_1b.sh
bash scripts/synid-sql/qwen3_to_qwen2.5_0.5b.sh
```

Default SynID-SQL hyperparameters:

- KD loss: `csd`
- KD ratio: `0.7`
- contrastive weights: `alpha=0.3`, `beta=0.3`
- pooling temperature: `5`
- contrastive temperature: `0.05`
- syntax weighting: enabled
- syntax lambda: `2.0`
- Qwen3 layer pair: student `27`, teacher `35`
- Llama3 layer pair: student `15`, teacher `31`
- Qwen3 to Qwen2.5 layer pair: `-1` to `-1`

## Inference and Evaluation

Inference runs five reporting seeds by default:

```text
10, 42, 50, 100, 1234
```

It evaluates on:

- `spider_data:test`, max new tokens `856`
- `spider_syn:test`, max new tokens `756`
- `spider_realistic:test`, max new tokens `755`
- `spider_dk:test`, max new tokens `663`

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

bash scripts/student_sft/infer_qwen2.5_0.5b.sh
bash scripts/student_sft/eval_qwen2.5_0.5b.sh
```

SynID-SQL:

```bash
bash scripts/synid-sql/infer_qwen3_to_qwen3_0.6b.sh
bash scripts/synid-sql/eval_qwen3_to_qwen3_0.6b.sh

bash scripts/synid-sql/infer_llama3_to_llama3_1b.sh
bash scripts/synid-sql/eval_llama3_to_llama3_1b.sh

bash scripts/synid-sql/infer_qwen3_to_qwen2.5_0.5b.sh
bash scripts/synid-sql/eval_qwen3_to_qwen2.5_0.5b.sh
```

Override seeds if needed:

```bash
export INFER_SEEDS=10,42,50,100,1234
```

## Tests

Run all tests:

```bash
python -m pytest tests
```

For a quick check of the SynID augmentation components:

```bash
python -m pytest tests/test_synid_augmentation.py
```

## Notes

- Training and vLLM augmentation are intended for CUDA Linux/WSL.
- Windows can be used for lightweight formatting and unit tests, but not for
  the full DeepSpeed/vLLM training workflow.
- Hugging Face gated models such as Llama require prior access and login.
