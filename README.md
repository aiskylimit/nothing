# Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models

## Setup
Create the environment:

```sh
mamba env create -f environment.yml
```

Activate it before running the code:

```sh
mamba activate pad
```

## Dataset
This codebase is configured to train directly from the Hugging Face dataset:

```text
pvdhihihi/ultra-feedback
```

The active training config is:

```text
training_configs/gemma-2-2b-it-pd.yaml
```

The relevant dataset settings are:

```yaml
dataset_mixer:
  pvdhihihi/ultra-feedback: 1.0
dataset_splits:
- train_prefs
- test_prefs
local_dataset: false
```

If the dataset is private or gated, log in to Hugging Face first:

```sh
huggingface-cli login
```

If `pvdhihihi/ultra-feedback` uses `train` and `test` instead of `train_prefs` and `test_prefs`, update the same config file to:

```yaml
dataset_splits:
- train
- test
```

## Training
After setup, run training directly:

```sh
bash run_ppd.sh
```

The trained model will be saved under:

```text
outputs/*
```

Before launching a long run, check these paths in `training_configs/gemma-2-2b-it-pd.yaml`:

- `model_name_or_path`: local path or Hugging Face id for the student model.
- `dataset_mixer`: dataset id to train on.
- `output_dir`: where checkpoints and final outputs are written.

## Optional: Regenerate Local PAD Data
The original repository generated a local PAD dataset before training. This is no longer required for the current Hugging Face dataset workflow.

Only run these commands if you want to recreate the original local generated dataset:

```sh
bash data_gen/scripts/sampling.sh
bash data_gen/scripts/pipeline_n4_gemma.sh
```

If you switch back to a locally generated dataset, also update `training_configs/gemma-2-2b-it-pd.yaml` so `dataset_mixer` points to the local dataset directory and `local_dataset` is `true`.

## Evaluation

We follow the official implementation for evaluation on AlpacaEval 2, Arena-Hard, MT-Bench and GSM8K.

* AlpacaEval 2: Please refer to the [AlpacaEval repo](https://github.com/tatsu-lab/alpaca_eval) for evaluation.

* Arena-Hard: Please refer to to the [Arena-Hard-Auto repo](https://github.com/lm-sys/arena-hard-auto) for evaluation.

* MT-Bench: Please refer to the [FastChat repo](https://github.com/lm-sys/FastChat) for evaluation.

* GSM8K: Please refer to the [ZeroEval repo](https://github.com/WildEval/ZeroEval) for evaluation.


## Training Report

The report below is from the original Gemma PAD experiment and is kept as a reference.

### Overview
This part contains training logs and comparative analysis of three preference alignment methods: SimPO, DPO, and PAD. We document the training process, implementation details, and performance metrics for each approach.

### Implementations
- **DPO**: Based on the implementation from [TRL](https://github.com/huggingface/trl)
- **SimPO**: Based on the implementation from [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)

### Training Configuration

#### Models
- **Student Model**: Gemma-2-2B-It
- **Teacher Model**: Gemma-2-9B-It

#### Hardware
- **GPUs**: 2 × A800 (80G)

#### Training Parameters
- **Training Type**: Full parameter fine-tuning
- **Memory Optimization**: ZeRO Stage 2
- **Epochs**: 1
- **Precision**: BFloat16
- **Dataset Size**:
  - Training samples: 55,321
  - Test samples: 1,130
- **Batch Size**: 128
- **Total Training Steps**: 432
- **Maximum Sequence Length**: 2048
- **Per Device Train Batch Size**: 2
- **Per Device Evaluation Batch Size**: 2
- **Gradient Accumulation Steps**: 32
- **Evaluation Frequency**: Every 100 training steps
- **Gradient Checkpointing**: Enabled

For additional parameters, please refer to the paper or the configuration files.

### Results

| Method | GPU Hours | Alpaca-Eval 2.0 LC (%) |
|--------|-----------|---------------------------------|
| DPO    | 8.7856    | 43.77                           |
| SimPO  | 7.2672    | 44.94                           |
| PAD    | 7.2884    | 45.73                           |

You can find the training log under `gemma-log/*`.

#### Analysis
- **Training Efficiency**: PAD and SimPO require similar computational resources, while DPO demands notably more. This efficiency difference is primarily because DPO requires loading an additional reference model during training, whereas PAD and SimPO do not.
- **Performance**: PAD outperforms both SimPO and DPO in terms of win rate, which aligns with the findings reported in the submission paper.
