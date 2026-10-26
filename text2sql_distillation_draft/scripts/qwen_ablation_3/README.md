# Qwen Ablation 3

This folder trains on generated LoRA-218 data only. The training data is copied
from:

`processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen/train.jsonl`

The prepared training directory intentionally does not include
`teacher_train_0.bin/.idx`, so SynID-SQL uses the same `train_0` inputs for the
student and teacher forward passes.

Variants:

- `sft`: student-only full-parameter SFT (`--type lm`, no teacher, no LoRA). Batch size 4, gradient accumulation 4, lr 5e-5, max length 1612 / prompt 1479, run on 2 GPUs (`GPUS_PER_JOB=2 RUNNER_GPU_LIST=0,1`).
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
ABLATION_SET=sft bash scripts/qwen_ablation_3/run_all.sh
```

Skip data preparation if the target directory already exists:

```bash
SKIP_PREPARE=true bash scripts/qwen_ablation_3/run_all.sh
```
