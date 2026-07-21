# synid_ce_no_keywords_weight_lora_436

SynID-CE CSD sweep on Spider privileged LoRA-436 augmented data without SQL
keyword/schema token up-weighting during semantic-anchor pooling.

Processed data:

- `processed_data/benchmarks/spider_data/synid_privileged_lora_436/qwen`

Defaults:

- `KD_RATIO=0.7`
- `SYNID_ALPHA_BETAS=(0.01 0.05)`
- `SYNID_POOL_TAU=5`
- `SYNID_CONTRASTIVE_TAUS=(0.05 0.01)`
- `SYNID_USE_SYNTAX_WEIGHTS=false`
- `SYNID_SYNTAX_LAMBDA=1.0`

Grid:

| Script | ID | Config | Student layers | Teacher layers | KD ratio | Contrastive tau | Alpha | Beta |
|---|---|---|---|---|---:|---:|---:|---:|
| `train_g01.sh` | G01 | last1 | `27` | `35` | 0.7 | 0.05 | 0.01 | 0.01 |
| `train_g02.sh` | G02 | last3 | `25,26,27` | `33,34,35` | 0.7 | 0.05 | 0.01 | 0.01 |
| `train_g03.sh` | G03 | last1 | `27` | `35` | 0.7 | 0.01 | 0.01 | 0.01 |
| `train_g04.sh` | G04 | last3 | `25,26,27` | `33,34,35` | 0.7 | 0.01 | 0.01 | 0.01 |
| `train_g05.sh` | G05 | last1 | `27` | `35` | 0.7 | 0.05 | 0.05 | 0.05 |
| `train_g06.sh` | G06 | last3 | `25,26,27` | `33,34,35` | 0.7 | 0.05 | 0.05 | 0.05 |
| `train_g07.sh` | G07 | last1 | `27` | `35` | 0.7 | 0.01 | 0.05 | 0.05 |
| `train_g08.sh` | G08 | last3 | `25,26,27` | `33,34,35` | 0.7 | 0.01 | 0.05 | 0.05 |

The default wrapper runs G01 through G08.

```bash
bash scripts/qwen_updated_2/synid_ce_no_keywords_weight_lora_436/run_full_pipeline.sh
```
