# synid_ce_multilayer_3

SynID-CE multilayer CSD sweep on Spider privileged data. Train entrypoints are
named `train_g*.sh`, and the run wrapper filters only that prefix so utility
scripts are not scheduled.

This grid distills the final 3 consecutive decoder layers of both models:

- student `Qwen/Qwen3-0.6B`: `25,26,27`
- teacher `Qwen/Qwen3-4B-Instruct-2507`: `33,34,35`

Fixed:

- `SYNID_KD_LOSS=csd`
- `SYNID_BETA=0.1`
- `SYNID_CONTRASTIVE_TAU=0.05`
- `SYNID_POOLING=sc`
- `MAX_LENGTH=2048`
- `MAX_PROMPT_LENGTH=1536`
- `T_MAX_LENGTH=2048`
- `T_MAX_PROMPT_LENGTH=1800`

Grid:

| Script | ID | k | Student layers | Teacher layers | alpha | kd ratio |
|---|---|---:|---|---|---:|---:|
| `train_g01.sh` | G01 | 3 | `25,26,27` | `33,34,35` | 0.3 | 0.7 |
| `train_g02.sh` | G02 | 3 | `25,26,27` | `33,34,35` | 0.3 | 0.6 |
| `train_g03.sh` | G03 | 3 | `25,26,27` | `33,34,35` | 0.3 | 0.5 |
| `train_g04.sh` | G04 | 3 | `25,26,27` | `33,34,35` | 0.3 | 0.8 |
| `train_g05.sh` | G05 | 3 | `25,26,27` | `33,34,35` | 0.3 | 0.9 |
| `train_g06.sh` | G06 | 3 | `25,26,27` | `33,34,35` | 0.5 | 0.7 |
| `train_g07.sh` | G07 | 3 | `25,26,27` | `33,34,35` | 0.5 | 0.6 |
| `train_g08.sh` | G08 | 3 | `25,26,27` | `33,34,35` | 0.5 | 0.5 |
| `train_g09.sh` | G09 | 3 | `25,26,27` | `33,34,35` | 0.5 | 0.8 |
| `train_g10.sh` | G10 | 3 | `25,26,27` | `33,34,35` | 0.5 | 0.9 |
| `train_g11.sh` | G11 | 3 | `25,26,27` | `33,34,35` | 0.7 | 0.7 |
| `train_g12.sh` | G12 | 3 | `25,26,27` | `33,34,35` | 0.7 | 0.6 |
| `train_g13.sh` | G13 | 3 | `25,26,27` | `33,34,35` | 0.7 | 0.5 |
| `train_g14.sh` | G14 | 3 | `25,26,27` | `33,34,35` | 0.7 | 0.8 |
| `train_g15.sh` | G15 | 3 | `25,26,27` | `33,34,35` | 0.7 | 0.9 |
| `train_g16.sh` | G16 | 3 | `25,26,27` | `33,34,35` | 1 | 0.7 |
| `train_g17.sh` | G17 | 3 | `25,26,27` | `33,34,35` | 1 | 0.6 |
| `train_g18.sh` | G18 | 3 | `25,26,27` | `33,34,35` | 1 | 0.5 |
| `train_g19.sh` | G19 | 3 | `25,26,27` | `33,34,35` | 1 | 0.8 |
| `train_g20.sh` | G20 | 3 | `25,26,27` | `33,34,35` | 1 | 0.9 |

Run all 20 jobs sequentially on two GPUs, then infer 5 seeds on all 4 Spider
variants after each checkpoint:

```bash
bash scripts/qwen/synid_ce_multilayer_3/run_full_grid_multiseed.sh
```

Run the full pipeline in one command: train, infer, format, then evaluate:

```bash
bash scripts/qwen/synid_ce_multilayer_3/run_full_pipeline.sh
```

By default inference selects the checkpoint with the best dev `exact_match`
from each training `log.txt`. The selection record is written to:

```text
run_logs/<timestamp>/checkpoint_selection.jsonl
```

Use `INFER_CHECKPOINT_METRIC=latest` to force the latest-checkpoint behavior.

The wrapper expands to:

```bash
INFER_SEEDS=10,42,50,100,1234 \
FORMAT_AFTER_INFER=true \
SKIP_EXISTING=true \
INFER_CHECKPOINT_METRIC=exact_match \
bash running.sh \
  --mode sequential \
  --gpus 0,1 \
  --gpus-per-job 2 \
  --skip-finalize \
  --filter scripts/qwen/synid_ce_multilayer_3/train_g \
  --infer-after-train \
  --infer-script scripts/qwen/synid_ce_multilayer_3/infer_multiseed.py \
  --infer-benchmarks spider_data,spider_syn,spider_realistic,spider_dk \
  --infer-split test \
  --infer-db full \
  --infer-batch-size 100 \
  --infer-output-root results/infer/synid_ce_multilayer_3 \
  --infer-extra-args "--flush-every 100"
```

Run one config through `running.sh`:

```bash
INFER_SEEDS=10,42,50,100,1234 \
FORMAT_AFTER_INFER=true \
SKIP_EXISTING=true \
bash running.sh \
  --mode sequential \
  --gpus 0,1 \
  --gpus-per-job 2 \
  --skip-finalize \
  --filter scripts/qwen/synid_ce_multilayer_3/train_g02.sh \
  --infer-after-train \
  --infer-script scripts/qwen/synid_ce_multilayer_3/infer_multiseed.py \
  --infer-benchmarks spider_data,spider_syn,spider_realistic,spider_dk \
  --infer-split test \
  --infer-db full \
  --infer-batch-size 100 \
  --infer-output-root results/infer/synid_ce_multilayer_3 \
  --infer-extra-args "--flush-every 100"
```

Multi-seed inference output layout:

```text
results/infer/synid_ce_multilayer_3/qwen/<benchmark>/seed<seed>/<run>__ckpt<step>__test__full_sql_result.json
```

Format and evaluate all generated multi-seed outputs:

```bash
bash scripts/qwen/synid_ce_multilayer_3/format_eval_multiseed.sh
```

Evaluation defaults to `--etype all`, so the Spider evaluator reports both
exact-match and execution metrics.

After evaluation, per-seed JSON summaries are written to:

```text
results/eval/synid_ce_multilayer_3/qwen/seed<seed>/eval_grid_results.json
results/eval/synid_ce_multilayer_3/qwen/seed<seed>/best_grid_by_benchmark.json
results/eval/synid_ce_multilayer_3/qwen/seed<seed>/best_grid_overall.json
```

`eval_grid_results.json` stores one item per grid and benchmark. Each item has
one `scores` field:

```json
{
  "scores": {
    "Exact Match": 0.742,
    "Execution Accuracy": 0.701
  }
}
```

The checkpoint used for inference is stored as the final artifact field:

```json
{
  "artifacts": {
    "run_name": "train_g01__ckpt1090__test",
    "pred_path": "...pred.sql",
    "gold_path": "...gold.sql",
    "eval_log_path": "...etype-all.timeout-60.log",
    "ckpt_path": "results/qwen3/.../1090"
  }
}
```

Best selection uses `Execution Accuracy` as the primary metric and `Exact Match`
as the tie-breaker. The overall best requires all four Spider variants.

Evaluate one benchmark only:

```bash
bash scripts/qwen/synid_ce_multilayer_3/format_eval_multiseed.sh spider_dk
```
