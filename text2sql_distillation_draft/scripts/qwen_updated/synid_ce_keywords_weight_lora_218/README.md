# synid_ce_keywords_weight_lora_218

SynID-CE CSD sweep on Spider privileged LoRA-218 augmented data with SQL
keyword/schema token up-weighting during semantic-anchor pooling.

Processed data:

- `processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen`

Defaults:

- `KD_RATIO=0.7`
- `SYNID_ALPHA=0.1`
- `SYNID_BETA=0.1`
- `SYNID_POOL_TAU=5`
- `SYNID_CONTRASTIVE_TAU=0.05`
- `SYNID_USE_SYNTAX_WEIGHTS=true`
- `SYNID_SYNTAX_LAMBDA=2.0`

Grid:

| Script | ID | Config | Student layers | Teacher layers | KD ratio |
|---|---|---|---|---|---:|
| `train_g01.sh` | G01 | last1 | `27` | `35` | 0.7 |
| `train_g02.sh` | G02 | last3 | `25,26,27` | `33,34,35` | 0.7 |

The default wrapper runs G01 and G02.

```bash
bash scripts/qwen_updated/synid_ce_keywords_weight_lora_218/run_full_pipeline.sh
```
