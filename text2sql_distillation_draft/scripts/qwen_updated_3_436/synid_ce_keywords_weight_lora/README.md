# synid_ce_keywords_weight_lora

SynID-CE CSD alpha/beta sweep on Spider privileged LoRA-436 augmented data with SQL keyword/schema token up-weighting during semantic-anchor pooling. Alpha and beta use the same value in each grid item.

Processed data:

- `processed_data/benchmarks/spider_data/synid_privileged_lora_436/qwen`

Teacher PEFT checkpoint:

- `hf://distillation-sql/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/436`

Defaults:

- `KD_RATIO=0.7`
- `BATCH_SIZE=8`
- `GRAD_ACC=4`
- `RUNNER_GPU_LIST=0`
- `GPUS_PER_JOB=1`
- `SYNID_POOL_TAU=5`
- `SYNID_CONTRASTIVE_TAU=0.05` and `0.01`
- `SYNID_STUDENT_LAYERS=27` for `last1`; `25,26,27` for `last3`
- `SYNID_TEACHER_LAYERS=35` for `last1`; `33,34,35` for `last3`
- `SYNID_USE_SYNTAX_WEIGHTS=true`
- `SYNID_SYNTAX_LAMBDA=2.0`

Grid:

| Script | ID | Layer config | Student layers | Teacher layers | Tau | SYNID_ALPHA | SYNID_BETA |
|---|---|---|---|---|---:|---:|---:|
| `train_g01.sh` | G01 | last1 | `27` | `35` | 0.05 | 0.1 | 0.1 |
| `train_g02.sh` | G02 | last1 | `27` | `35` | 0.05 | 0.2 | 0.2 |
| `train_g03.sh` | G03 | last1 | `27` | `35` | 0.05 | 0.3 | 0.3 |
| `train_g04.sh` | G04 | last1 | `27` | `35` | 0.05 | 0.4 | 0.4 |
| `train_g05.sh` | G05 | last1 | `27` | `35` | 0.05 | 0.5 | 0.5 |
| `train_g06.sh` | G06 | last3 | `25,26,27` | `33,34,35` | 0.05 | 0.1 | 0.1 |
| `train_g07.sh` | G07 | last3 | `25,26,27` | `33,34,35` | 0.05 | 0.2 | 0.2 |
| `train_g08.sh` | G08 | last3 | `25,26,27` | `33,34,35` | 0.05 | 0.3 | 0.3 |
| `train_g09.sh` | G09 | last3 | `25,26,27` | `33,34,35` | 0.05 | 0.4 | 0.4 |
| `train_g10.sh` | G10 | last3 | `25,26,27` | `33,34,35` | 0.05 | 0.5 | 0.5 |
| `train_g11.sh` | G11 | last1 | `27` | `35` | 0.01 | 0.1 | 0.1 |
| `train_g12.sh` | G12 | last1 | `27` | `35` | 0.01 | 0.2 | 0.2 |
| `train_g13.sh` | G13 | last1 | `27` | `35` | 0.01 | 0.3 | 0.3 |
| `train_g14.sh` | G14 | last1 | `27` | `35` | 0.01 | 0.4 | 0.4 |
| `train_g15.sh` | G15 | last1 | `27` | `35` | 0.01 | 0.5 | 0.5 |
| `train_g16.sh` | G16 | last3 | `25,26,27` | `33,34,35` | 0.01 | 0.1 | 0.1 |
| `train_g17.sh` | G17 | last3 | `25,26,27` | `33,34,35` | 0.01 | 0.2 | 0.2 |
| `train_g18.sh` | G18 | last3 | `25,26,27` | `33,34,35` | 0.01 | 0.3 | 0.3 |
| `train_g19.sh` | G19 | last3 | `25,26,27` | `33,34,35` | 0.01 | 0.4 | 0.4 |
| `train_g20.sh` | G20 | last3 | `25,26,27` | `33,34,35` | 0.01 | 0.5 | 0.5 |

The default wrapper runs G01 through G20.

```bash
bash scripts/qwen_updated_3_436/synid_ce_keywords_weight_lora/run_full_pipeline.sh
```
