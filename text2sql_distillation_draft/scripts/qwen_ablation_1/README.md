# Qwen Ablation 1

This folder runs the first SynID contrastive-loss ablation using the exact
hyperparameter setting from:

`scripts/qwen_updated_2/synid_ce_keywords_weight_lora_218/train_g01.sh`

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
- `SYNID_USE_SYNTAX_WEIGHTS=true`
- `SYNID_SYNTAX_LAMBDA=2.0`

Ablation scripts:

- `train_no_con1.sh`: disables `l_con1`
- `train_no_con2.sh`: disables `l_con2`
- `train_no_con1_no_con2.sh`: disables both

Run all three:

```bash
bash scripts/qwen_ablation_1/run_all.sh
```

Run one:

```bash
bash scripts/qwen_ablation_1/train_no_con1.sh
```

Train + infer + format/eval:

```bash
bash scripts/qwen_ablation_1/run_full_pipeline.sh
```

Dry-run:

```bash
bash scripts/qwen_ablation_1/run_all.sh --dry-run
```
