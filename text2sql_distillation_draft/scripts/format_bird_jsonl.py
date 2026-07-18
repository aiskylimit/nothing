#!/usr/bin/env python3
"""Format BIRD benchmark JSON files into prompt/response JSONL.

The output matches the existing Spider-style `format_data/*.jsonl` shape:
each row contains `system_prompt`, `user_prompt`, and `response`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BIRD_ROOT = Path("benchmarks_2/bird")
PROMPT_DIR = Path("prompts/single_turn/bird_generator")
SPLIT_CONFIG = {
    "train": ("train/train.json", "train/train_tables.json"),
    "val": ("dev/dev.json", "dev/dev_tables.json"),
    "test": ("dev/dev.json", "dev/dev_tables.json"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def serialize_schema_entry(entry: dict[str, Any]) -> str:
    table_names = entry.get("table_names_original") or entry.get("table_names") or []
    column_names = entry.get("column_names_original") or entry.get("column_names") or []
    foreign_keys = entry.get("foreign_keys") or []

    table_to_columns: dict[int, list[str]] = {idx: [] for idx in range(len(table_names))}
    for table_idx, col_name in column_names:
        if table_idx == -1:
            continue
        table_to_columns.setdefault(table_idx, []).append(str(col_name))

    table_lines = []
    for idx, table_name in enumerate(table_names):
        cols = ", ".join(table_to_columns.get(idx, []))
        table_lines.append(f"- {table_name}({cols})")

    fk_lines = []
    for src_idx, dst_idx in foreign_keys:
        src_table_idx, src_col = column_names[src_idx]
        dst_table_idx, dst_col = column_names[dst_idx]
        src_table = table_names[src_table_idx]
        dst_table = table_names[dst_table_idx]
        fk_lines.append(f"- {src_table}.{src_col} -> {dst_table}.{dst_col}")

    schema_parts = ["Tables:"] + table_lines
    if fk_lines:
        schema_parts.extend(["", "Foreign keys:"] + fk_lines)
    return "\n".join(schema_parts)


def build_schema_lookup(tables_path: Path) -> dict[str, str]:
    return {entry["db_id"]: serialize_schema_entry(entry) for entry in read_json(tables_path)}


def make_record(
    sample: dict[str, Any],
    schema_lookup: dict[str, str],
    system_prompt: str,
    user_prompt_template: str,
) -> dict[str, str]:
    db_id = str(sample["db_id"]).strip()
    question = str(sample["question"]).strip()
    evidence = str(sample.get("evidence") or "").strip()
    sql = str(sample["SQL"]).strip()
    schema = schema_lookup[db_id]

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt_template.format(
            question=question,
            evidence=evidence,
            schema=schema,
        ),
        "response": json.dumps({"sql": sql}, ensure_ascii=False),
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_split(root: Path, prompt_dir: Path, split: str) -> Path:
    if split not in SPLIT_CONFIG:
        available = ", ".join(sorted(SPLIT_CONFIG))
        raise ValueError(f"Unsupported split {split!r}. Available splits: {available}")

    data_rel, tables_rel = SPLIT_CONFIG[split]
    data_path = root / data_rel
    tables_path = root / tables_rel
    output_path = root / "format_data" / f"{split}.jsonl"

    system_prompt = read_text(prompt_dir / "system_prompt.txt")
    user_prompt_template = read_text(prompt_dir / "user_prompt.txt")
    samples = read_json(data_path)
    schema_lookup = build_schema_lookup(tables_path)
    rows = [make_record(sample, schema_lookup, system_prompt, user_prompt_template) for sample in samples]
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BIRD_ROOT)
    parser.add_argument("--prompt-dir", type=Path, default=PROMPT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    for split in args.splits:
        format_split(args.root, args.prompt_dir, split)


if __name__ == "__main__":
    main()
