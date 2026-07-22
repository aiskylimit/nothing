# Llama Baseline KD Pipeline

This folder runs Llama Spider KD baselines end to end:

1. finetune baseline adapters,
2. run multi-seed inference,
3. format predictions,
4. evaluate Spider variants,
5. upload artifacts to Hugging Face.

Teacher default:

```bash
hf://distillation-sql/llama_spider/llama/sft_sft_llama3_8b_lora_spider_lm_e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/e5-bs2-lr0.0001-G8-N2-NN1-lora-16-64-0.1/1090
```

Student default:

```bash
meta-llama/Llama-3.2-1B-Instruct
```

## Baselines

```text
train_fkl.sh       --type fkl
train_rkl.sh       --type rkl
train_sfkl.sh      --type sfkl
train_srkl.sh      --type srkl
train_csd.sh       --type csd
train_distillm.sh  --type adaptive-srkl
train_amid.sh      --type adaptive-amid
train_fdd.sh       finetuning/fdd_finetune.py + --type sfkl + layer mapping
```

FDD is intentionally different from the other baselines. It uses
`finetuning/fdd_finetune.py`, keeps a normal KD divergence as the base loss
(`FDD_DISTILL_TYPE=sfkl` by default), and adds hidden-trajectory loss with
Llama layer mapping:

```bash
STUDENT_LAYER_MAPPING_OVERRIDE="13 16"
TEACHER_LAYER_MAPPING_OVERRIDE="29 32"
```

## Run Everything

```bash
bash scripts/llama/baselines/run_full_pipeline.sh
```

Default inference seeds:

```text
10,42,50,100,1234
```

Default eval benchmarks:

```text
spider_data,spider_syn,spider_realistic,spider_dk
```

Outputs:

```text
results/llama_baselines/
results/infer/llama_baselines/llama/
results/eval/llama_baselines/llama/
```

## Common Overrides

```bash
RUNNER_GPU_LIST=0,1 GPUS_PER_JOB=2 \
INFER_SEEDS=10,42,50,100,1234 \
KD_RATIO=0.7 \
bash scripts/llama/baselines/run_full_pipeline.sh
```

Run one baseline:

```bash
bash running.sh \
  --mode sequential \
  --gpus 0,1 \
  --gpus-per-job 2 \
  --skip-finalize \
  --filter scripts/llama/baselines/train_csd.sh \
  --infer-after-train \
  --infer-script scripts/llama/baselines/infer_multiseed.py \
  --infer-benchmarks spider_data,spider_syn,spider_realistic,spider_dk \
  --infer-split test \
  --infer-db full \
  --infer-batch-size 32 \
  --infer-output-root results/infer/llama_baselines
```

Skip upload:

```bash
SKIP_HF_UPLOAD=1 bash scripts/llama/baselines/run_full_pipeline.sh
```

Upload target defaults to:

```bash
HF_REPO_ID=distillation-sql/llama_spider
HF_REPO_TYPE=model
```

The upload step requires `HF_TOKEN` with write access.
