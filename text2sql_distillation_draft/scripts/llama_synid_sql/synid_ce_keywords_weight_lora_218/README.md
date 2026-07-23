# synid_ce_keywords_weight_lora_218

Llama SynID-SQL training on `synid_privileged_lora_218`.

Defaults:

- `DATA_DIR=processed_data/benchmarks/spider_data/synid_privileged_lora_218/llama`
- `TEACHER_PEFT_PATH=https://huggingface.co/distillation-sql/llama_spider/tree/main/llama/sft_sft_llama3_8b_lora_spider_lm_e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/1090`
- `BATCH_SIZE=8`
- `GRAD_ACC=4`
- `KD_RATIO=0.7`
- `SYNID_ALPHA=0.3`
- `SYNID_BETA=0.3`
- `SYNID_POOL_TAU=5`
- `SYNID_CONTRASTIVE_TAU=0.05`
- `SYNID_USE_SYNTAX_WEIGHTS=true`
- `SYNID_SYNTAX_LAMBDA=2.0`

Grid:

| Script | ID | Config | Student layers | Teacher layers | KD ratio |
| --- | --- | --- | --- | --- | --- |
| `train_g01.sh` | G01 | last1, alpha=beta=0.3 | 15 | 31 | 0.7 |
| `train_g02.sh` | G02 | last1, alpha=beta=0.1 | 15 | 31 | 0.7 |

Run:

```bash
RUN_GPUS=0 bash scripts/llama_synid_sql/synid_ce_keywords_weight_lora_218/train_g01.sh
RUN_GPUS=0 bash scripts/llama_synid_sql/synid_ce_keywords_weight_lora_218/train_g02.sh
```
