# synid_ce_no_keywords_weight

SynID-CE CSD sweep on Spider privileged data without SQL keyword/schema token
up-weighting during semantic-anchor pooling.

Fixed:

- `SYNID_KD_LOSS=csd`
- `SYNID_BETA=0.1`
- `SYNID_CONTRASTIVE_TAU=0.05`
- `SYNID_POOLING=sc`
- `SYNID_USE_SYNTAX_WEIGHTS=false`
- `SYNID_SYNTAX_LAMBDA=1.0`
- `MAX_LENGTH=2048`
- `MAX_PROMPT_LENGTH=1536`
- `T_MAX_LENGTH=2048`
- `T_MAX_PROMPT_LENGTH=1800`

Grid:

| Script | ID | k | Student layers | Teacher layers | alpha | kd ratio |
|---|---|---:|---|---|---:|---:|
| `train_g01.sh` | G01 | 1 | `27` | `35` | 0.3 | 0.7 |
| `train_g02.sh` | G02 | 1 | `27` | `35` | 0.3 | 0.6 |
| `train_g03.sh` | G03 | 1 | `27` | `35` | 0.5 | 0.7 |
| `train_g04.sh` | G04 | 1 | `27` | `35` | 0.5 | 0.6 |
| `train_g05.sh` | G05 | 1 | `27` | `35` | 1 | 0.7 |
| `train_g06.sh` | G06 | 1 | `27` | `35` | 1 | 0.6 |
| `train_g07.sh` | G07 | 3 | `25,26,27` | `33,34,35` | 0.3 | 0.7 |
| `train_g08.sh` | G08 | 3 | `25,26,27` | `33,34,35` | 0.3 | 0.6 |
| `train_g09.sh` | G09 | 3 | `25,26,27` | `33,34,35` | 0.5 | 0.7 |
| `train_g10.sh` | G10 | 3 | `25,26,27` | `33,34,35` | 0.5 | 0.6 |
| `train_g11.sh` | G11 | 3 | `25,26,27` | `33,34,35` | 1 | 0.7 |
| `train_g12.sh` | G12 | 3 | `25,26,27` | `33,34,35` | 1 | 0.6 |

Run full train, multi-seed infer, format, and eval:

```bash
bash scripts/qwen/synid_ce_no_keywords_weight/run_full_pipeline.sh
```

Memory-light version:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUNNER_GPU_LIST=0,1 \
GPUS_PER_JOB=2 \
BATCH_SIZE=8 \
GRAD_ACC=1 \
EVAL_BATCH_SIZE=1 \
INFER_BATCH_SIZE=16 \
bash scripts/qwen/synid_ce_no_keywords_weight/run_full_pipeline.sh
```

Inference output:

```text
results/infer/synid_ce_no_keywords_weight/qwen/<benchmark>/seed<seed>/<run>__ckpt<step>__test__full_sql_result.json
```

Evaluation summaries:

```text
results/eval/synid_ce_no_keywords_weight/qwen/seed<seed>/eval_grid_results.json
results/eval/synid_ce_no_keywords_weight/qwen/seed<seed>/best_grid_by_benchmark.json
results/eval/synid_ce_no_keywords_weight/qwen/seed<seed>/best_grid_overall.json
```