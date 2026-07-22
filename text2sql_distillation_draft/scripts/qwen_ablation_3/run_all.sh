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
INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"
FORMAT_AFTER_INFER="${FORMAT_AFTER_INFER:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
INFER_BENCHMARKS="${INFER_BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-100}"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-results/infer/qwen_ablation_3}"
INFER_FLUSH_EVERY="${INFER_FLUSH_EVERY:-100}"
INFER_CHECKPOINT_METRIC="${INFER_CHECKPOINT_METRIC:-exact_match}"
ABLATION_SET="${ABLATION_SET:-csd,distillm,synid_sql}"
SKIP_PREPARE="${SKIP_PREPARE:-false}"

is_dry_run=false
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    is_dry_run=true
  fi
done

if ! [[ "${SKIP_PREPARE}" =~ ^(1|true|yes|y)$ ]] && [[ "${is_dry_run}" != "true" ]]; then
  bash scripts/qwen_ablation_3/prepare_data.sh
fi

export INFER_SEEDS
export FORMAT_AFTER_INFER
export SKIP_EXISTING

script_for_variant() {
  case "$1" in
    csd)
      echo "scripts/qwen_ablation_3/train_csd_generated_lora218.sh"
      ;;
    distillm|adaptive_srkl|adaptive-srkl)
      echo "scripts/qwen_ablation_3/train_distillm_generated_lora218.sh"
      ;;
    synid_sql|synid)
      echo "scripts/qwen_ablation_3/train_synid_sql_generated_lora218.sh"
      ;;
    *)
      echo "Unknown ablation variant: $1" >&2
      echo "Supported variants: csd,distillm,synid_sql" >&2
      exit 2
      ;;
  esac
}

canonical_variant() {
  case "$1" in
    distillm|adaptive_srkl|adaptive-srkl)
      echo "distillm"
      ;;
    synid_sql|synid)
      echo "synid_sql"
      ;;
    *)
      echo "$1"
      ;;
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
    --log-dir "run_logs/qwen_ablation_3/${variant_name}" \
    --infer-after-train \
    --infer-script scripts/qwen_updated_2/synid_ce_keywords_weight_lora_218/infer_multiseed.py \
    --infer-benchmarks "${INFER_BENCHMARKS}" \
    --infer-split test \
    --infer-db full \
    --infer-batch-size "${INFER_BATCH_SIZE}" \
    --infer-output-root "${INFER_OUTPUT_ROOT}/${variant_name}" \
    --infer-checkpoint-metric "${INFER_CHECKPOINT_METRIC}" \
    --infer-extra-args "--flush-every ${INFER_FLUSH_EVERY}" \
    "$@"
done
