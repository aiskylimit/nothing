# synid_ce_keywords_weight_lora_218

Qwen2.5 SynID-SQL training on Qwen shared-tokenizer `synid_privileged_lora_218` data.

Defaults:

- `DATA_DIR=processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen`
- `CKPT=Qwen/Qwen2.5-0.5B-Instruct`
- `TEACHER_CKPT=Qwen/Qwen3-4B-Instruct-2507`
- `TEACHER_PEFT_PATH=hf://Dream-AI-HUST/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090`
- `BATCH_SIZE=4`
- `GRAD_ACC=4`
- `KD_RATIO=0.7`
- `SYNID_STUDENT_LAYERS=-1`
- `SYNID_TEACHER_LAYERS=-1`
- `SYNID_LAYER_CONFIG=k1_lm_head_s-1_t-1`
- `SYNID_USE_SYNTAX_WEIGHTS=true`
- `SYNID_SYNTAX_LAMBDA=2.0`

Grid:

| Script | ID | alpha=beta | contrastive tau | Student layers | Teacher layers | KD ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `train_g01.sh` | G01 | 0.3 | 0.05 | -1 | -1 | 0.7 |
| `train_g02.sh` | G02 | 0.1 | 0.05 | -1 | -1 | 0.7 |
| `train_g03.sh` | G03 | 0.3 | 0.01 | -1 | -1 | 0.7 |
| `train_g04.sh` | G04 | 0.1 | 0.01 | -1 | -1 | 0.7 |
| `train_g05.sh` | G05 | 0.3 | 0.03 | -1 | -1 | 0.7 |
| `train_g06.sh` | G06 | 0.1 | 0.03 | -1 | -1 | 0.7 |
| `train_g07.sh` | G07 | 0.3 | 0.1 | -1 | -1 | 0.7 |
| `train_g08.sh` | G08 | 0.1 | 0.1 | -1 | -1 | 0.7 |

Run:

```bash
RUN_GPUS=0 bash scripts/qwen2.5/synid_sql/synid_ce_keywords_weight_lora_218/train_g01.sh
RUN_GPUS=0 bash scripts/qwen2.5/synid_sql/synid_ce_keywords_weight_lora_218/train_g02.sh
RUN_GPUS=0 bash scripts/qwen2.5/synid_sql/synid_ce_keywords_weight_lora_218/train_g03.sh
RUN_GPUS=0 bash scripts/qwen2.5/synid_sql/synid_ce_keywords_weight_lora_218/train_g04.sh
RUN_GPUS=0 bash scripts/qwen2.5/synid_sql/synid_ce_keywords_weight_lora_218/train_g05.sh
RUN_GPUS=0 bash scripts/qwen2.5/synid_sql/synid_ce_keywords_weight_lora_218/train_g06.sh
RUN_GPUS=0 bash scripts/qwen2.5/synid_sql/synid_ce_keywords_weight_lora_218/train_g07.sh
RUN_GPUS=0 bash scripts/qwen2.5/synid_sql/synid_ce_keywords_weight_lora_218/train_g08.sh
```
