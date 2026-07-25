# Qwen Ablation 4

Computational-overhead runs for the dual-contrastive representation loss.
Each run uses the same memory profile as the corresponding `kd_2` baseline:
`csd` uses batch size 2 with gradient accumulation 8, and `distillm` uses
batch size 1 with gradient accumulation 16. Runs stop after 50 optimizer steps.
Overhead metrics are logged per optimizer step and then averaged by the collector.

Methods:

- `distillm`: DistiLLM/adaptive SRKL baseline.
- `distillm_dcr`: DistiLLM/adaptive SRKL with DCR.
- `csd`: CSD baseline.
- `csd_dcr`: CSD with DCR.

Run all:

```bash
bash scripts/qwen_ablation_4/run_full_pipeline.sh
```

Run one:

```bash
ABLATION_SET=csd_dcr bash scripts/qwen_ablation_4/run_all.sh
```

Collect existing logs:

```bash
python scripts/qwen_ablation_4/collect_overhead_results.py
```

Outputs:

- `results/overhead/qwen_ablation_4/computational_overhead.json`
- `results/overhead/qwen_ablation_4/computational_overhead_table.tex`
