#!/usr/bin/env python3
"""Convert Spider benchmark files to the JSON schema consumed by KID training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a SQL Query Generator for Spider-style text-to-SQL tasks.\n\n"
    "Your task is to generate a valid SQLite SQL query that answers the user's "
    "natural language question using only the provided database schema.\n\n"
    "Rules:\n"
    "- Use only tables, columns, and relationships that exist in the provided schema.\n"
    "- Do not invent schema elements.\n"
    "- Generate a single executable SQL query.\n"
    "- Return only the columns needed to answer the question.\n"
    "- Use DISTINCT when duplicates are possible and the question implies unique results.\n"
    "- If the question requires aggregation, sorting, grouping, filtering, or limiting, use the correct SQL clauses.\n"
    "- Use JOINs only when needed, and use schema-consistent join keys.\n"
    "- Qualify column names when they may be ambiguous.\n"
    "- Prefer concise, conventional SQLite SQL.\n"
    "- Ensure the query is syntactically valid.\n\n"
    "Output format:\n"
    "{\n"
    '  "sql": "The complete SQL query"\n'
    "}\n\n"
    "Return only the JSON object and nothing else."
)

USER_PROMPT_TEMPLATE = (
    "QUESTION:\n"
    "{question}\n\n"
    "SCHEMA:\n"
    "{schema}\n\n"
    "Generate a valid SQLite SQL query that answers the question using only the provided schema.\n"
    "Return only the JSON object in the required format."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_schema_entry(entry: dict[str, Any]) -> str:
    table_names = entry.get("table_names_original") or entry.get("table_names") or []
    column_names = entry.get("column_names_original") or entry.get("column_names") or []
    foreign_keys = entry.get("foreign_keys") or []

    table_to_columns: dict[int, list[str]] = {idx: [] for idx in range(len(table_names))}
    for table_idx, column_name in column_names:
        if table_idx == -1:
            continue
        table_to_columns.setdefault(table_idx, []).append(str(column_name))

    table_lines = []
    for table_idx, table_name in enumerate(table_names):
        columns = ", ".join(table_to_columns.get(table_idx, []))
        table_lines.append(f"- {table_name}({columns})")

    fk_lines = []
    for source_idx, target_idx in foreign_keys:
        source_table_idx, source_column = column_names[source_idx]
        target_table_idx, target_column = column_names[target_idx]
        source_table = table_names[source_table_idx]
        target_table = table_names[target_table_idx]
        fk_lines.append(f"- {source_table}.{source_column} -> {target_table}.{target_column}")

    parts = ["Tables:"] + table_lines
    if fk_lines:
        parts.extend(["", "Foreign keys:"] + fk_lines)
    return "\n".join(parts)


def build_schema_lookup(tables_path: Path) -> dict[str, str]:
    return {
        str(entry["db_id"]): serialize_schema_entry(entry)
        for entry in read_json(tables_path)
    }


def make_record(sample: dict[str, Any], schema_lookup: dict[str, str]) -> dict[str, Any]:
    db_id = str(sample["db_id"]).strip()
    question = str(sample["question"]).strip()
    sql = str(sample["query"]).strip()
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question, schema=schema_lookup[db_id])

    return {
        "instruction": user_prompt,
        "input": "",
        "output": json.dumps({"sql": sql}, ensure_ascii=False),
        "system": SYSTEM_PROMPT,
        "db_id": db_id,
        "question": question,
    }


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_dataset_info(path: Path) -> None:
    dataset_info = {
        "example_text2sql_train": {
            "file_name": "example_text2sql_train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
            },
        },
        "example_text2sql_dev": {
            "file_name": "example_text2sql_dev.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
            },
        },
    }
    path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spider-root", type=Path, default=Path("../../benchmarks/spider_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("dbgpt_hub/data/spider_benchmarks_codes"))
    parser.add_argument("--include-train-others", action="store_true")
    args = parser.parse_args()

    schema_lookup = build_schema_lookup(args.spider_root / "tables.json")

    train_samples = read_json(args.spider_root / "train_spider.json")
    if args.include_train_others:
        train_samples = train_samples + read_json(args.spider_root / "train_others.json")
    dev_samples = read_json(args.spider_root / "dev.json")

    train_rows = [make_record(sample, schema_lookup) for sample in train_samples]
    dev_rows = [make_record(sample, schema_lookup) for sample in dev_samples]

    write_json(args.output_dir / "example_text2sql_train.json", train_rows)
    write_json(args.output_dir / "example_text2sql_dev.json", dev_rows)
    write_dataset_info(args.output_dir / "dataset_info.json")

    print(f"Wrote {len(train_rows)} train rows to {args.output_dir / 'example_text2sql_train.json'}")
    print(f"Wrote {len(dev_rows)} dev rows to {args.output_dir / 'example_text2sql_dev.json'}")


if __name__ == "__main__":
    main()
