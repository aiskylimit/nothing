#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-processed_data/benchmarks/spider_data/synid_privileged_lora_218/qwen}"
TARGET_DATA_DIR="${TARGET_DATA_DIR:-processed_data/benchmarks/spider_data/generated_lora_218_train_only/qwen}"

required_source_files=(
  "train.jsonl"
  "train_0.bin"
  "train_0.idx"
  "valid.jsonl"
  "valid_0.bin"
  "valid_0.idx"
)

for file in "${required_source_files[@]}"; do
  if [[ ! -s "${SOURCE_DATA_DIR}/${file}" ]]; then
    echo "[prepare-data] missing or empty source file: ${SOURCE_DATA_DIR}/${file}" >&2
    exit 2
  fi
done

mkdir -p "${TARGET_DATA_DIR}"

copy_if_present() {
  local file="$1"
  if [[ -s "${SOURCE_DATA_DIR}/${file}" ]]; then
    cp -f "${SOURCE_DATA_DIR}/${file}" "${TARGET_DATA_DIR}/${file}"
  fi
}

copy_if_present "train.jsonl"
copy_if_present "train_0.bin"
copy_if_present "train_0.idx"
copy_if_present "valid.jsonl"
copy_if_present "valid_0.bin"
copy_if_present "valid_0.idx"
copy_if_present "test.jsonl"
copy_if_present "test_0.bin"
copy_if_present "test_0.idx"

rm -f \
  "${TARGET_DATA_DIR}/teacher_train.jsonl" \
  "${TARGET_DATA_DIR}/teacher_train_0.bin" \
  "${TARGET_DATA_DIR}/teacher_train_0.idx"

echo "[prepare-data] source: ${SOURCE_DATA_DIR}"
echo "[prepare-data] target: ${TARGET_DATA_DIR}"
echo "[prepare-data] copied generated train data only; teacher_train artifacts removed"
