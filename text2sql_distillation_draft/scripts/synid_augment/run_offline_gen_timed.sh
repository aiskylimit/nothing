#! /usr/bin/env bash
# Run every offline SynID data-generation stage as its own process and log
# wall-clock seconds per stage into $RUN_DIR/stage_timings.jsonl, then summarize.
#
# Stages (timed separately, summed by summarize_gen_timing.py):
#   setup_download_teacher  fetch teacher LoRA adapter into the HF cache (setup, not counted)
#   prep_format             scripts/format_spider_jsonl.py (student train.jsonl)
#   gen_loops               scripts/synid_augment/run_spider_aug_loops_v2.py
#   select_merge            scripts/synid_augment/overall_symthetic.py
#   build_train             scripts/synid_augment/build_teacher_train_from_final_merged.py
#   tokenize_train          process_data.py --split train

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to a new output directory}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:?Set TEACHER_PEFT_PATH (e.g. hf://.../218)}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
TOKENIZE_MODEL="${TOKENIZE_MODEL:-Qwen/Qwen3-0.6B}"
SPIDER_SRC="${SPIDER_SRC:-datasets/train/spider_data}"
DB_ROOT="${DB_ROOT:-benchmarks/spider_data/database}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.9}"
NUM_LOOPS="${NUM_LOOPS:-5}"

if [[ -e "${RUN_DIR}/stage_timings.jsonl" ]]; then
  echo "[error] ${RUN_DIR}/stage_timings.jsonl already exists; use a fresh RUN_DIR" >&2
  exit 2
fi

SPIDER_ROOT="${RUN_DIR}/spider_root"
GEN_DIR="${RUN_DIR}/synid_aug_v2_lora_218"
SYNID_DIR="${RUN_DIR}/synid_privileged_lora_218/qwen"
PROCESSED_DIR="${RUN_DIR}/processed_data"
LOG_DIR="${RUN_DIR}/logs"
STAGE_LOG="${RUN_DIR}/stage_timings.jsonl"
mkdir -p "${SPIDER_ROOT}" "${SYNID_DIR}" "${LOG_DIR}" "${RUN_DIR}/tmp" "${RUN_DIR}/pylibs"

# Keep every temp/cache write off the small root filesystem.
export TMPDIR="${RUN_DIR}/tmp"

# Isolated copy (symlinks) of the Spider train inputs so format_data lands in RUN_DIR.
for f in train_spider.json tables.json; do
  ln -sfn "$(readlink -f "${SPIDER_SRC}/${f}")" "${SPIDER_ROOT}/${f}"
done

# exec_eval needs sqlparse; expose it without modifying the python environment.
if ! "${PYTHON}" -c "import sqlparse" 2>/dev/null; then
  SQLPARSE_DIR="${SQLPARSE_DIR:-${REPO_ROOT}/.venv/lib/python3.10/site-packages/sqlparse}"
  ln -sfn "${SQLPARSE_DIR}" "${RUN_DIR}/pylibs/sqlparse"
fi
export PYTHONPATH="${REPO_ROOT}:${RUN_DIR}/pylibs${PYTHONPATH:+:${PYTHONPATH}}"

run_stage() {
  local name="$1"
  shift
  local t0 t1 rc secs started finished
  started="$(date -Is)"
  echo "[stage:${name}] start ${started}"
  t0="$(date +%s.%N)"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  rc="${PIPESTATUS[0]}"
  t1="$(date +%s.%N)"
  finished="$(date -Is)"
  secs="$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.3f", b - a }')"
  printf '{"stage": "%s", "seconds": %s, "returncode": %s, "started_at": "%s", "finished_at": "%s"}\n' \
    "${name}" "${secs}" "${rc}" "${started}" "${finished}" >> "${STAGE_LOG}"
  echo "[stage:${name}] done rc=${rc} seconds=${secs}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] stage ${name} failed (rc=${rc}); stopping" >&2
    "${PYTHON}" "${SCRIPT_DIR}/summarize_gen_timing.py" --run-dir "${RUN_DIR}" || true
    exit "${rc}"
  fi
}

run_stage setup_download_teacher \
  "${PYTHON}" -c "from infer import _resolve_ckpt_dir; print(_resolve_ckpt_dir('${TEACHER_PEFT_PATH}'))"

run_stage prep_format \
  "${PYTHON}" scripts/format_spider_jsonl.py --root "${SPIDER_ROOT}" --splits train

run_stage gen_loops \
  "${PYTHON}" scripts/synid_augment/run_spider_aug_loops_v2.py \
    --benchmark spider \
    --root "${SPIDER_ROOT}" \
    --output-root "${GEN_DIR}" \
    --db-root "${DB_ROOT}" \
    --model "${MODEL}" \
    --teacher-peft-path "${TEACHER_PEFT_PATH}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --similarity-threshold "${SIMILARITY_THRESHOLD}" \
    --num-loops "${NUM_LOOPS}"

run_stage select_merge \
  "${PYTHON}" scripts/synid_augment/overall_symthetic.py --base-dir "${GEN_DIR}"

run_stage build_train bash -c "
  set -euo pipefail
  cp '${SPIDER_ROOT}/format_data/train.jsonl' '${SYNID_DIR}/train.jsonl'
  '${PYTHON}' scripts/synid_augment/build_teacher_train_from_final_merged.py \
    --input '${GEN_DIR}/final_merged.jsonl' \
    --output '${SYNID_DIR}/teacher_train.jsonl' \
    --train-output '${SYNID_DIR}/train.jsonl'
"

run_stage tokenize_train \
  "${PYTHON}" process_data.py \
    --model-path "${TOKENIZE_MODEL}" \
    --model-type qwen \
    --data-dir "${SYNID_DIR}" \
    --processed-data-dir "${PROCESSED_DIR}" \
    --split train \
    --max-length 2048 \
    --max-prompt-length 1536 \
    --t-max-length 2048 \
    --t-max-prompt-length 1800 \
    --data-process-workers 8

"${PYTHON}" "${SCRIPT_DIR}/summarize_gen_timing.py" --run-dir "${RUN_DIR}"
touch "${RUN_DIR}/DONE"
