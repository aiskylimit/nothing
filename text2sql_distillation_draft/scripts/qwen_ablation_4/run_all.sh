#! /usr/bin/env bash

set -euo pipefail

if [[ -n "${RUN_MASTER_PORT:-}" && "${ALLOW_RUNNING_SH_UTILITY:-0}" != "1" ]]; then
  echo "[skip] run_all.sh is a wrapper; use it directly, not as a running.sh job."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_MODE="${RUN_MODE:-sequential}"
RUNNER_GPU_LIST="${RUNNER_GPU_LIST:-0,1}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
ABLATION_SET="${ABLATION_SET:-distillm,distillm_dcr,csd,csd_dcr}"

export SKIP_EXISTING

script_for_variant() {
  case "$1" in
    distillm)
      echo "scripts/qwen_ablation_4/train_distillm.sh"
      ;;
    distillm_dcr|distillm-dcr)
      echo "scripts/qwen_ablation_4/train_distillm_dcr.sh"
      ;;
    csd)
      echo "scripts/qwen_ablation_4/train_csd.sh"
      ;;
    csd_dcr|csd-dcr)
      echo "scripts/qwen_ablation_4/train_csd_dcr.sh"
      ;;
    *)
      echo "Unknown ablation variant: $1" >&2
      echo "Supported variants: distillm,distillm_dcr,csd,csd_dcr" >&2
      exit 2
      ;;
  esac
}

canonical_variant() {
  case "$1" in
    distillm-dcr) echo "distillm_dcr" ;;
    csd-dcr) echo "csd_dcr" ;;
    *) echo "$1" ;;
  esac
}

IFS=', ' read -r -a VARIANTS <<< "${ABLATION_SET}"
for variant in "${VARIANTS[@]}"; do
  if [[ -z "${variant}" ]]; then
    continue
  fi

  train_script="$(script_for_variant "${variant}")"
  variant_name="$(canonical_variant "${variant}")"

  bash running.sh \
    --mode "${RUN_MODE}" \
    --gpus "${RUNNER_GPU_LIST}" \
    --gpus-per-job "${GPUS_PER_JOB}" \
    --skip-finalize \
    --filter "${train_script}" \
    --log-dir "run_logs/qwen_ablation_4/${variant_name}" \
    "$@"
done
