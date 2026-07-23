#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python}"
INFER_SCRIPT="${INFER_SCRIPT:-scripts/qwen_updated_2/synid_ce_keywords_weight_lora_218/infer_multiseed.py}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
CKPT_STEP="${CKPT_STEP:-1090}"
DEFAULT_CKPT_PATH="results/qwen3/qwen_ablation_3_train_synid_sql_generated_lora218_spider_synid_generated-lora218-train-only-e5-bs8-lr0.0001-G4-N1-NN1-kd0.7-csd-tau0.05-a0.3-b0.3-k1_last_s27_t35-sl27-tl35-poolsc-keywords-lambda2.0-same-teacher-input-lora-16-64-0.1/${CKPT_STEP}"

RUN_GPUS="${RUN_GPUS:-0}"
INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"
INFER_BENCHMARKS="${INFER_BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-100}"
INFER_SPLIT="${INFER_SPLIT:-test}"
INFER_DB="${INFER_DB:-full}"
INFER_FLUSH_EVERY="${INFER_FLUSH_EVERY:-100}"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-results/infer/qwen_ablation_3/synid_sql/qwen_ablation_3}"

export INFER_SEEDS
export SKIP_EXISTING="${SKIP_EXISTING:-false}"
export FORMAT_AFTER_INFER="${FORMAT_AFTER_INFER:-true}"
export CUDA_VISIBLE_DEVICES="${RUN_GPUS}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

safe_name() {
  local raw="$1"
  raw="${raw//[^[:alnum:]._-]/_}"
  printf '%s' "${raw}"
}

max_length_for() {
  case "$1:${INFER_SPLIT}" in
    spider_data:test) printf '856' ;;
    spider_syn:test) printf '756' ;;
    spider_realistic:test) printf '755' ;;
    spider_dk:test) printf '663' ;;
    *) printf '1024' ;;
  esac
}

resolve_ckpt_path() {
  if [[ -n "${CKPT_PATH:-}" ]]; then
    printf '%s' "${CKPT_PATH}"
    return
  fi

  if [[ -d "${DEFAULT_CKPT_PATH}" ]]; then
    printf '%s' "${DEFAULT_CKPT_PATH}"
    return
  fi

  mapfile -t matches < <(
    find results/qwen3 \
      -path "*qwen_ablation_3_train_synid_sql_generated_lora218*/${CKPT_STEP}" \
      -type d 2>/dev/null | sort
  )

  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "No qwen_ablation_3 synid_sql checkpoint ${CKPT_STEP} found under results/qwen3." >&2
    echo "Expected default: ${DEFAULT_CKPT_PATH}" >&2
    echo "Set CKPT_PATH=/path/to/${CKPT_STEP} and rerun." >&2
    exit 2
  fi

  printf '%s' "${matches[$((${#matches[@]} - 1))]}"
}

run_benchmark() {
  local benchmark="$1"
  local ckpt_path="$2"
  local benchmark_name db_name output_dir output_path max_length

  benchmark_name="$(safe_name "${benchmark}")"
  db_name="$(safe_name "${INFER_DB}")"
  output_dir="${INFER_OUTPUT_ROOT}/${benchmark_name}"
  output_path="${output_dir}/qwen_ablation_3_train_synid_sql_generated_lora218__ckpt${CKPT_STEP}__${INFER_SPLIT}__${db_name}_sql_result.json"
  max_length="$(max_length_for "${benchmark}")"

  mkdir -p "${output_dir}"
  echo "[infer] benchmark=${benchmark} max_length=${max_length}"
  echo "        output=${output_path}"

  "${PYTHON_BIN}" "${INFER_SCRIPT}" \
    --benchmark "${benchmark}" \
    --split "${INFER_SPLIT}" \
    --db "${INFER_DB}" \
    --model "${MODEL_PATH}" \
    --ckpt_path "${ckpt_path}" \
    --device cuda \
    --batch-size "${INFER_BATCH_SIZE}" \
    --max-length "${max_length}" \
    --output_path "${output_path}" \
    --flush-every "${INFER_FLUSH_EVERY}"
}

main() {
  local ckpt_path benchmark

  ckpt_path="$(resolve_ckpt_path)"
  echo "[infer] qwen_ablation_3 synid_sql"
  echo "        ckpt=${ckpt_path}"
  echo "        gpus=${CUDA_VISIBLE_DEVICES}"
  echo "        seeds=${INFER_SEEDS}"
  echo "        output_root=${INFER_OUTPUT_ROOT}"

  IFS=', ' read -r -a benchmarks <<< "${INFER_BENCHMARKS}"
  for benchmark in "${benchmarks[@]}"; do
    [[ -z "${benchmark}" ]] && continue
    run_benchmark "${benchmark}" "${ckpt_path}"
  done

  echo "[infer-done]"
}

main "$@"
