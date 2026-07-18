# Llama SFT Scripts

These scripts train Spider/SynID SFT with `finetuning/finetune.py --type lm`.

Context lengths are set from the Llama 1B token-length scan on `train.jsonl`:

- `MAX_PROMPT_LENGTH=1536` for prompt max `1496`
- `MAX_LENGTH=1664` for prompt + response max `1621`

Process Llama-tokenized data first:

```bash
bash scripts/llama/sft/process_llama3_1b_data.sh
```

This writes to `processed_data/spider_data/llama`.

If you want to regenerate with the 8B tokenizer explicitly:

```bash
bash scripts/llama/sft/process_llama3_8b_data.sh
```

Run 8B LoRA SFT:

```bash
RUN_GPUS=0,1 bash scripts/llama/sft/sft_llama3_8b_lora.sh
```

Run 1B full SFT:

```bash
RUN_GPUS=0 bash scripts/llama/sft/sft_llama3_1b.sh
```

Run train -> multi-seed inference -> eval:

```bash
bash scripts/llama/sft/run_train_infer_eval.sh
```

Run only one model:

```bash
TARGET=8b bash scripts/llama/sft/run_train_infer_eval.sh
TARGET=1b bash scripts/llama/sft/run_train_infer_eval.sh
```

Inference outputs go under `results/infer/llama_sft/llama`.
Eval logs and per-seed JSON summaries go under `results/eval/llama_sft/llama`.

Useful overrides:

```bash
SEED=43 RUN_GPUS=0,1 BATCH_SIZE=2 GRAD_ACC=8 LR=0.0001 EPOCHS=5 \
  bash scripts/llama/sft/sft_llama3_8b_lora.sh
```

Default outputs go under `results/llama/`.
