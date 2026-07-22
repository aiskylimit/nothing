# Qwen Ablation 3

This folder trains on generated LoRA-218 data only. The training data is copied
from:

`processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen/train.jsonl`

The prepared training directory intentionally does not include
`teacher_train_0.bin/.idx`, so SynID-SQL uses the same `train_0` inputs for the
student and teacher forward passes.

Variants:

- `csd`: standard CSD KD, following `scripts/kd_2/csd`.
- `distillm`: DistillM/adaptive SRKL, following `scripts/kd_2/distillm`.
- `synid_sql`: SynID-SQL full contrastive objective on train-only generated data.

Prepare data:

```bash
bash scripts/qwen_ablation_3/prepare_data.sh
```

Train all variants:

```bash
bash scripts/qwen_ablation_3/run_all.sh
```

Train + infer + format/eval:

```bash
bash scripts/qwen_ablation_3/run_full_pipeline.sh
```

Run one variant:

```bash
ABLATION_SET=csd bash scripts/qwen_ablation_3/run_all.sh
```

Skip data preparation if the target directory already exists:

```bash
SKIP_PREPARE=true bash scripts/qwen_ablation_3/run_all.sh
```
