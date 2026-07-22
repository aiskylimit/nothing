#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

echo "[data] download Dream-AI-HUST/sql_benchmarks"
hf download Dream-AI-HUST/sql_benchmarks \
  --repo-type dataset \
  --local-dir .

if [[ -f benchmarks.zip ]]; then
  echo "[data] unzip benchmarks.zip"
  unzip -o benchmarks.zip
else
  echo "[data] missing benchmarks.zip after download" >&2
  exit 2
fi

if [[ -f data.zip ]]; then
  echo "[data] unzip data.zip"
  unzip -o data.zip
else
  echo "[data] missing data.zip after download" >&2
  exit 2
fi

GPU_LIST="${LLAMA_SYNID_GPU_LIST:-${RUNNER_GPU_LIST:-0,1,2}}"
IFS=', ' read -r -a GPUS <<< "${GPU_LIST}"
INFER_BENCHMARKS="${INFER_BENCHMARKS:-spider_data,spider_syn,spider_realistic,spider_dk}"
INFER_SEEDS="${INFER_SEEDS:-10,42,50,100,1234}"
FORMAT_AFTER_INFER="${FORMAT_AFTER_INFER:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"
INFER_FLUSH_EVERY="${INFER_FLUSH_EVERY:-100}"
INFER_CHECKPOINT_METRIC="${INFER_CHECKPOINT_METRIC:-exact_match}"
INFER_OUTPUT_ROOT_BASE="${INFER_OUTPUT_ROOT_BASE:-results/infer}"
INFER_OUTPUT_ROOT_FOR_EVAL="${INFER_OUTPUT_ROOT_FOR_EVAL:-${INFER_OUTPUT_ROOT_BASE}/llama_synid_sql}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/eval/llama_synid_sql}"

FOLDERS=(
  "synid_ce_keywords_weight_lora"
  "synid_ce_keywords_weight_lora_218"
  "synid_ce_keywords_weight_lora_436"
)

if (( ${#GPUS[@]} < ${#FOLDERS[@]} )); then
  echo "Need at least ${#FOLDERS[@]} GPUs, got ${#GPUS[@]} from GPU_LIST=${GPU_LIST}" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-run_logs/llama_synid_sql/${timestamp}}"
mkdir -p "${LOG_DIR}"

echo "[llama-synid] logs: ${LOG_DIR}"
echo "[llama-synid] gpu list: ${GPU_LIST}"
echo "[llama-synid] batch size: ${BATCH_SIZE:-8}"
echo "[llama-synid] grad acc: ${GRAD_ACC:-4}"
echo "[llama-synid] infer seeds: ${INFER_SEEDS}"
echo "[llama-synid] infer benchmarks: ${INFER_BENCHMARKS}"

pids=()
names=()
logs=()

launch_folder() {
  local folder="$1"
  local gpu="$2"
  local train_filter="scripts/llama_synid_sql/${folder}/train_g"
  local log_file="${LOG_DIR}/${folder}.log"

  if [[ ! -f "scripts/llama_synid_sql/${folder}/train_g01.sh" || ! -f "scripts/llama_synid_sql/${folder}/train_g02.sh" ]]; then
    echo "Missing train_g01.sh or train_g02.sh under scripts/llama_synid_sql/${folder}" >&2
    exit 2
  fi

  echo "[llama-synid] launch ${folder} on GPU ${gpu}"
  (
    INFER_SEEDS="${INFER_SEEDS}" \
    FORMAT_AFTER_INFER="${FORMAT_AFTER_INFER}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    BATCH_SIZE="${BATCH_SIZE:-8}" \
    GRAD_ACC="${GRAD_ACC:-4}" \
    bash running.sh \
      --mode sequential \
      --gpus "${gpu}" \
      --gpus-per-job 1 \
      --skip-finalize \
      --filter "${train_filter}" \
      --log-dir "${LOG_DIR}/running/${folder}" \
      --infer-after-train \
      --infer-script scripts/llama/baselines/infer_multiseed.py \
      --infer-benchmarks "${INFER_BENCHMARKS}" \
      --infer-split test \
      --infer-db full \
      --infer-batch-size "${INFER_BATCH_SIZE}" \
      --infer-output-root "${INFER_OUTPUT_ROOT_BASE}" \
      --infer-checkpoint-metric "${INFER_CHECKPOINT_METRIC}" \
      --infer-extra-args "--flush-every ${INFER_FLUSH_EVERY}"
  ) > "${log_file}" 2>&1 &

  pids+=("$!")
  names+=("${folder}")
  logs+=("${log_file}")
}

for idx in "${!FOLDERS[@]}"; do
  launch_folder "${FOLDERS[$idx]}" "${GPUS[$idx]}"
done

status=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "[llama-synid] done ${names[$idx]} -> ${logs[$idx]}"
  else
    exit_code="$?"
    echo "[llama-synid] failed ${names[$idx]} exit=${exit_code} log=${logs[$idx]}" >&2
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[llama-synid] one or more jobs failed" >&2
  exit "${status}"
fi

echo "[llama-synid] train + infer finished"

echo "[llama-synid] format + eval"
IFS=',' read -r -a EVAL_BENCHMARK_ARGS <<< "${INFER_BENCHMARKS}"
for i in "${!EVAL_BENCHMARK_ARGS[@]}"; do
  EVAL_BENCHMARK_ARGS[$i]="${EVAL_BENCHMARK_ARGS[$i]//[[:space:]]/}"
done
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT_FOR_EVAL}" \
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
COLLECT_SCRIPT="${COLLECT_SCRIPT:-scripts/llama/baselines/collect_eval_results.py}" \
  bash scripts/llama/baselines/format_eval_multiseed.sh "${EVAL_BENCHMARK_ARGS[@]}"

case "${SKIP_HF_UPLOAD:-0}" in
  1|true|TRUE|yes|YES|y|Y)
    echo "[upload-skip] SKIP_HF_UPLOAD=${SKIP_HF_UPLOAD}"
    ;;
  *)
    echo "[llama-synid] upload to Hugging Face"
    "${PYTHON:-python}" scripts/llama_synid_sql/upload_to_hf.py
    ;;
esac

echo "[llama-synid] full pipeline done"
