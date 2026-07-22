# Qwen Ablation 2

This folder runs the second SynID-SQL ablation using the same fixed setting as
`scripts/qwen_ablation_1`, but removes syntax-aware weighting only.

Fixed G01 setting:

- `KD_RATIO=0.7`
- `BATCH_SIZE=8`
- `GRAD_ACC=4`
- `SYNID_STUDENT_LAYERS=27`
- `SYNID_TEACHER_LAYERS=35`
- `SYNID_CONTRASTIVE_TAU=0.05`
- `SYNID_ALPHA=0.3`
- `SYNID_BETA=0.3`
- `SYNID_KD_LOSS=csd`
- `SYNID_USE_CON1=true`
- `SYNID_USE_CON2=true`
- `SYNID_USE_SYNTAX_WEIGHTS=false`
- `SYNID_SYNTAX_LAMBDA=1.0`

Run:

```bash
bash scripts/qwen_ablation_2/run_all.sh
```

Train + infer + format/eval:

```bash
bash scripts/qwen_ablation_2/run_full_pipeline.sh
```

Dry-run:

```bash
bash scripts/qwen_ablation_2/run_all.sh --dry-run
```
