#!/usr/bin/env bash
# Does the purified privilege vector carry CONTENT or only view FORMAT?
#
# Runs scripts/math/pvsd_vector_probe.py across the three candidate injection
# layers (L/4, L/3, L/2), plus a no-PIE control, and prints one summary table.
# Each run steers with (a) the correct reference, (b) a mismatched reference
# purified the same way, (c) a random vector of equal norm, and reports the change
# in log P(gold solution).
#
#   bash scripts/math/probe_pvsd.sh
#   VIEWS=full_solution,partial_solution,answer_only bash scripts/math/probe_pvsd.sh
#
# No training. A few minutes on one GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
# shellcheck source=../remote/paths.sh
source "${REPO_ROOT}/scripts/remote/paths.sh"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
OUT_DIR="${OUT_DIR:-${PVSD_RESULTS_ROOT}/pvsd/probe}"
NUM_EXAMPLES="${NUM_EXAMPLES:-8}"
NUM_CORRUPT="${NUM_CORRUPT:-2}"
VIEWS="${VIEWS:-full_solution}"
ALPHAS="${ALPHAS:-0.5,1.0,2.0,4.0}"
TOP_K_HEADS="${TOP_K_HEADS:-10}"
PIE_LAYERS="${PIE_LAYERS:-all}"
PIE_HEAD_CHUNK="${PIE_HEAD_CHUNK:-8}"
PIE_NUM_EXAMPLES="${PIE_NUM_EXAMPLES:-2}"
LAYER_FRACTIONS="${LAYER_FRACTIONS:-quarter third half}"
EXTRA_ARGS=("$@")

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

common=(
    --model_name "${MODEL_NAME}"
    --num_examples "${NUM_EXAMPLES}"
    --num_corrupt "${NUM_CORRUPT}"
    --views "${VIEWS}"
    --alphas "${ALPHAS}"
    --top_k_heads "${TOP_K_HEADS}"
    --pie_layers "${PIE_LAYERS}"
    --pie_head_chunk "${PIE_HEAD_CHUNK}"
    --pie_num_examples "${PIE_NUM_EXAMPLES}"
)

for fraction in ${LAYER_FRACTIONS}; do
    echo "=== probe: PIE heads, injection layer = ${fraction} of depth ==="
    python scripts/math/pvsd_vector_probe.py \
        "${common[@]}" \
        --pvsd_layer_fraction "${fraction}" \
        --output "${OUT_DIR}/pie_${fraction}.json" \
        "${EXTRA_ARGS[@]}"
done

# Control: same pipeline, but read every head of the injection layer instead of the
# PIE-selected ones. If this matches the PIE runs, localisation is not doing work.
echo "=== probe: control, no PIE (all heads of the injection layer) ==="
python scripts/math/pvsd_vector_probe.py \
    "${common[@]}" \
    --pvsd_layer_fraction quarter \
    --no_pie \
    --output "${OUT_DIR}/nopie_quarter.json" \
    "${EXTRA_ARGS[@]}"

python - "${OUT_DIR}" <<'SUMMARY'
import json, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
reports = sorted(out_dir.glob("*.json"))
if not reports:
    raise SystemExit(f"no probe reports in {out_dir}")

print("\n" + "=" * 96)
print("PROBE SUMMARY - mean delta log P(gold) vs no steering")
print("=" * 96)
header = f"{'run':<18}{'layer':>6}{'alpha':>8}{'correct':>12}{'mismatched':>12}{'random':>12}  verdict"
print(header)
print("-" * len(header))

for report_path in reports:
    report = json.loads(report_path.read_text())
    deltas = report["mean_gold_logprob_delta"]
    for alpha in report["alphas"]:
        row = {arm: deltas[f"alpha={alpha}/{arm}"] for arm in ("correct", "mismatched", "random")}
        baseline = max(row["mismatched"], row["random"])
        margin = row["correct"] - baseline
        if row["correct"] <= 0:
            verdict = "no effect"
        elif margin <= 0:
            verdict = "FORMAT ONLY"
        elif margin < 0.5 * abs(row["correct"]):
            verdict = "weak / mostly format"
        else:
            verdict = "content signal"
        print(
            f"{report_path.stem:<18}{report['injection_layer']:>6}{alpha:>8}"
            f"{row['correct']:>12.4f}{row['mismatched']:>12.4f}{row['random']:>12.4f}  {verdict}"
        )

print("\nPer-view template share (cos_raw_corrupt near 1.0 = the read-out is almost all template):")
for report_path in reports:
    report = json.loads(report_path.read_text())
    for view, stats in report["vector_diagnostics"].items():
        print(
            f"  {report_path.stem:<18}{view:<20}"
            f"cos_raw_corrupt={stats['cos_raw_corrupt']:.3f}  "
            f"transfer_ratio={stats['transfer_ratio']:.3f}  "
            f"cos(correct,mismatched)={stats['cos_correct_mismatched']:.3f}"
        )

print(
    "\nProceed to training only if 'correct' clearly beats both other arms at some alpha.\n"
    "If every row says FORMAT ONLY or no effect, training will distil noise."
)
SUMMARY
