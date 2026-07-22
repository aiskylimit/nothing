# Qwen Ablation 5

This folder trains SynID-SQL with the conventional training flow.

The default dataset is:

```bash
orig_processed_data/benchmarks/spider_data/qwen
```

The student and teacher forward passes use the same indexed `train_0` data by
default:

```bash
SYNID_USE_PRIVILEGED_TEACHER_INPUT=false
```

This ablation intentionally does not prepare or copy generated/privileged
training data. If `SYNID_USE_PRIVILEGED_TEACHER_INPUT=true` is passed, the train
script requires `teacher_train_0.bin/.idx` to exist in the same data directory.

Train:

```bash
bash scripts/qwen_ablation_5/run_all.sh
```

Train + infer + format/eval:

```bash
bash scripts/qwen_ablation_5/run_full_pipeline.sh
```

Dry-run:

```bash
bash scripts/qwen_ablation_5/run_all.sh --dry-run
```

Main setting:

- student: `Qwen/Qwen3-0.6B`
- teacher: `Qwen/Qwen3-4B-Instruct-2507`
- teacher LoRA: `hf://distillation-sql/baselines/qwen3/sft_sft_qwen3_4b_spider_lora/e5-bs4-lr0.0001-G4-N2-NN1-lora-32-64-0.1/1090`
- KD ratio: `0.7`
- SynID KD loss: `csd`
- SynID alpha/beta: `0.3/0.3`
- SynID contrastive tau: `0.05`
- SynID pooling: `sc`
- SynID syntax weights: `true`
- SynID layers: student `27`, teacher `35`
- LoRA: `r=16`, `alpha=64`, `dropout=0.1`
