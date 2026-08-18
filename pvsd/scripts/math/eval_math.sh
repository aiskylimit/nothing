#!/usr/bin/env bash
# Avg@8 on the math benchmarks for one or more checkpoints, then a summary table.
#
#   # one checkpoint, the three headline datasets
#   CHECKPOINTS=~/outputs/checkpoints/pvsd/qwen3_4b/qwen3_4b_pvsd/checkpoint-200 \
#     bash scripts/math/eval_math.sh
#
#   # several checkpoints / ablation arms at once
#   CHECKPOINTS="~/outputs/checkpoints/pvsd/qwen3_4b/qwen3_4b_pvsd_main/checkpoint-500
#                ~/outputs/checkpoints/pvsd/qwen3_4b/qwen3_4b_pvsd_no_purification/checkpoint-500" \
#     bash scripts/math/eval_math.sh
#
#   DATASETS="aime24 aime25 hmmt25 math500" bash scripts/math/eval_math.sh
#
# Metric: Avg@8 (val_n=8), matching the AVSD baseline table.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
# shellcheck source=../remote/paths.sh
source "${REPO_ROOT}/scripts/remote/paths.sh"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
DATASETS="${DATASETS:-aime24 aime25 hmmt25}"
VAL_N="${VAL_N:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
RESULTS_ROOT="${RESULTS_ROOT:-${PVSD_RESULTS_ROOT}/math}"
CHECKPOINTS="${CHECKPOINTS:-}"
EXTRA_ARGS=("$@")

if [[ -z "${CHECKPOINTS}" ]]; then
    echo "set CHECKPOINTS to one or more LoRA checkpoint directories, e.g." >&2
    echo "  CHECKPOINTS=~/outputs/checkpoints/pvsd/qwen3_4b/qwen3_4b_pvsd/checkpoint-200 bash $0" >&2
    exit 1
fi

cd "${REPO_ROOT}"

for checkpoint in ${CHECKPOINTS}; do
    if [[ ! -d "${checkpoint}" ]]; then
        echo "checkpoint directory not found: ${checkpoint}" >&2
        exit 1
    fi
    # ${RESULTS_ROOT}/<run name>_<checkpoint-N>/<dataset>.json
    tag="$(basename "$(dirname "${checkpoint}")")_$(basename "${checkpoint}")"
    for dataset in ${DATASETS}; do
        output="${RESULTS_ROOT}/${tag}/${dataset}.json"
        if [[ -f "${output}" && "${OVERWRITE:-0}" != "1" ]]; then
            echo "=== skip ${tag} / ${dataset} (exists; OVERWRITE=1 to redo) ==="
            continue
        fi
        echo "=== eval ${tag} / ${dataset} (Avg@${VAL_N}) ==="
        python -m pvsd.math.evaluate \
            --base_model "${BASE_MODEL}" \
            --checkpoint_dir "${checkpoint}" \
            --dataset "${dataset}" \
            --val_n "${VAL_N}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            --top_p "${TOP_P}" \
            --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
            --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
            --output_file "${output}" \
            "${EXTRA_ARGS[@]}"
    done
done

python - "${RESULTS_ROOT}" <<'SUMMARY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
datasets = []
for path in sorted(root.glob("*/*.json")):
    if path.name.endswith(".summary.json"):
        continue
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError:
        continue
    if "average_at_n_pct" not in report:
        continue
    dataset = report.get("dataset", path.stem)
    rows.setdefault(path.parent.name, {})[dataset] = report["average_at_n_pct"]
    if dataset not in datasets:
        datasets.append(dataset)

if not rows:
    raise SystemExit(f"no evaluation reports under {root}")

datasets.sort()
width = max(len(name) for name in rows) + 2
print("\n" + "=" * (width + 12 * len(datasets) + 8))
print("Avg@n (%) - higher is better")
print("=" * (width + 12 * len(datasets) + 8))
print("run".ljust(width) + "".join(d.rjust(12) for d in datasets) + "mean".rjust(11))
incomplete = False
for name, scores in sorted(rows.items()):
    values = [scores.get(d) for d in datasets]
    line = name.ljust(width)
    line += "".join(("-" if v is None else f"{v:.1f}").rjust(12) for v in values)
    # Only average complete rows: a mean over a subset of datasets is not comparable
    # with a mean over all of them.
    if all(v is not None for v in values):
        line += f"{sum(values) / len(values):.1f}".rjust(11)
    else:
        incomplete = True
        line += "incompl.".rjust(11)
    print(line)
if incomplete:
    print("\n'incompl.' = some datasets are missing for that run; finish them before comparing means.")
print("\nAVSD-reported Qwen3-4B baselines: Base 55.0, SFT 47.6, GRPO 56.6, OPSD 58.2, AVSD 59.9")
SUMMARY
