# Qwen Ablation 2.1

This folder runs the SynID contrastive ablation where the student prompt
representation is contrasted directly with the student response representation.
It uses the exact hyperparameter setting from:

`scripts/qwen_updated_2/synid_ce_keywords_weight_lora_218/train_g01.sh`

Fixed G01 setting:

- `KD_RATIO=0.7`
- `BATCH_SIZE=8`
- `GRAD_ACC=4`
- `SYNID_STUDENT_LAYERS=27`
- `SYNID_TEACHER_LAYERS=35`
- `SYNID_CONTRASTIVE_TAU=0.05`
- `SYNID_ALPHA=0.3`
- `SYNID_BETA=0.0`
- `SYNID_KD_LOSS=csd`
- `SYNID_USE_SYNTAX_WEIGHTS=true`
- `SYNID_SYNTAX_LAMBDA=2.0`
- `SYNID_CON1_POSITIVE_SOURCE=student_response`

Objective:

- `l_con1`: `student_prompt` vs `student_response`
- `l_con2`: disabled

Run:

```bash
bash scripts/qwen_ablation_2_1/train_prompt_vs_student_response.sh
```

Train + infer + format/eval:

```bash
bash scripts/qwen_ablation_2_1/run_full_pipeline.sh
```

Dry-run:

```bash
bash scripts/qwen_ablation_2_1/run_all.sh --dry-run
```
