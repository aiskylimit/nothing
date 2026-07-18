#!/usr/bin/env python3
"""Format Spider-style benchmark JSON files into prompt/response JSONL.

This writes files compatible with discover_context_length.ipynb:
each row contains system_prompt, user_prompt, and response.
"""

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

BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "spider_dk": {
        "splits": {"test": "test.json"},
        "tables": "tables.json",
        "question_fields": ("question",),
    },
    "spider_realistic": {
        "splits": {"test": "test.json"},
        "tables": "../spider_data/tables.json",
        "question_fields": ("question",),
    },
    "spider_syn": {
        "splits": {"test": "test.json"},
        "tables": "../spider_data/tables.json",
        "question_fields": ("SpiderSynQuestion", "question", "SpiderQuestion"),
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def get_question(sample: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = sample.get(field)
        if value:
            return str(value).strip()
    raise KeyError(f"missing question field; expected one of: {', '.join(fields)}")


def make_record(
    sample: dict[str, Any],
    schema_lookup: dict[str, str],
    question_fields: tuple[str, ...],
) -> dict[str, str]:
    db_id = sample["db_id"]
    question = get_question(sample, question_fields)
    schema = schema_lookup[db_id]
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question, schema=schema)
    response = json.dumps({"sql": str(sample["query"]).strip()}, ensure_ascii=False)

    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "response": response,
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_benchmark(root: Path, benchmark: str, split: str) -> Path:
    config = BENCHMARK_CONFIGS[benchmark]
    if split not in config["splits"]:
        available = ", ".join(sorted(config["splits"]))
        raise ValueError(f"{benchmark} does not include split {split!r}. Available splits: {available}")

    benchmark_dir = root / benchmark
    samples_path = benchmark_dir / config["splits"][split]
    tables_path = benchmark_dir / config["tables"]
    output_path = benchmark_dir / "format_data" / f"{split}.jsonl"

    samples = read_json(samples_path)
    schema_lookup = build_schema_lookup(tables_path)
    question_fields = tuple(config["question_fields"])
    rows = [make_record(sample, schema_lookup, question_fields) for sample in samples]
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmarks_2"))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=sorted(BENCHMARK_CONFIGS),
        choices=sorted(BENCHMARK_CONFIGS),
    )
    args = parser.parse_args()

    for benchmark in args.benchmarks:
        format_benchmark(args.root, benchmark, args.split)


if __name__ == "__main__":
    main()
