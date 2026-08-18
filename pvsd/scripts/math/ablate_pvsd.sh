#!/usr/bin/env bash
# The PVSD ablation matrix, run sequentially.
#
#   bash scripts/math/ablate_pvsd.sh                  # see the plan, run everything
#   DRY_RUN=1 bash scripts/math/ablate_pvsd.sh        # print the commands only
#   ONLY=main,no_purification bash scripts/math/ablate_pvsd.sh
#   SKIP=layer_third,layer_half bash scripts/math/ablate_pvsd.sh
#
# Every arm is one training run of the same script with one thing changed, so the
# comparison is matched-cost: the purification ablations still extract the corrupted
# contexts and differ only in which signal is injected.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="${SCRIPT_DIR}/train_pvsd_qwen3_4b.sh"

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PVSD_CKPT_ROOT:-${HOME}/outputs/checkpoints}/pvsd/qwen3_4b}"

# name | what it isolates | extra flags
ARMS=(
  "main|the method as specified|"
  "no_purification|steering without template subtraction|--pvsd_purification none"
  "template_only|the discarded component alone (should hurt)|--pvsd_purification template_only"
  "single_view|multi-view fusion vs one view|--pvsd_views full_solution"
  "frozen_calibration|online PIE vs one-time calibration|--pvsd_pie_every 0"
  "no_pie|PIE localisation vs all heads of the layer|--no_pvsd_pie_enabled"
  "layer_third|injection depth L/3|--pvsd_layer_fraction third"
  "layer_half|injection depth L/2|--pvsd_layer_fraction half"
  "alpha_0p5|steering strength 0.5|--pvsd_alpha 0.5"
  "alpha_2p0|steering strength 2.0|--pvsd_alpha 2.0"
  "heads_5|smaller head budget|--pvsd_top_k_heads 5"
  "heads_20|larger head budget|--pvsd_top_k_heads 20"
  "corrupt_cycle|donor choice: rotation instead of length matching|--pvsd_corrupt_match cycle"
  "jsd_beta0p5|JSD instead of reverse KL|--beta 0.5"
)

in_list() {
    local needle="$1" haystack="$2"
    [[ ",${haystack}," == *",${needle},"* ]]
}

selected=()
for arm in "${ARMS[@]}"; do
    name="${arm%%|*}"
    if [[ -n "${ONLY}" ]] && ! in_list "${name}" "${ONLY}"; then continue; fi
    if [[ -n "${SKIP}" ]] && in_list "${name}" "${SKIP}"; then continue; fi
    selected+=("${arm}")
done

if ((${#selected[@]} == 0)); then
    echo "no arms selected (ONLY='${ONLY}' SKIP='${SKIP}')" >&2
    exit 1
fi

echo "=== PVSD ablation plan (${#selected[@]} runs) ==="
for arm in "${selected[@]}"; do
    IFS='|' read -r name description extra <<<"${arm}"
    printf '  %-20s %s\n' "${name}" "${description}"
done
echo

for arm in "${selected[@]}"; do
    IFS='|' read -r name description extra <<<"${arm}"
    run_config="qwen3_4b_pvsd_${name}"
    echo "=================================================================="
    echo "ARM ${name}: ${description}"
    echo "  run_config=${run_config}"
    [[ -n "${extra}" ]] && echo "  flags: ${extra}"
    echo "=================================================================="

    # shellcheck disable=SC2206
    extra_args=(${extra})
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] RUN_CONFIG=${run_config} OUTPUT_DIR=${OUTPUT_ROOT} bash ${TRAIN} ${extra}"
        continue
    fi

    RUN_CONFIG="${run_config}" \
        OUTPUT_DIR="${OUTPUT_ROOT}" \
        bash "${TRAIN}" "${extra_args[@]}"
done

echo
echo "All selected arms finished. Checkpoints under ${OUTPUT_ROOT}/qwen3_4b_pvsd_<arm>/"
echo "Evaluate them with: CHECKPOINTS=... bash scripts/math/eval_math.sh"
