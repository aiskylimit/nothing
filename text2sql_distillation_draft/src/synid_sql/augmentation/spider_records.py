from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import read_json, read_text


JSON_SCHEMA = json.dumps({"sql": "The complete SQL query"}, indent=2)


def serialize_schema_entry(entry: dict[str, Any]) -> str:
    table_names = entry.get("table_names_original") or entry.get("table_names") or []
    column_names = entry.get("column_names_original") or entry.get("column_names") or []
    foreign_keys = entry.get("foreign_keys") or []

    table_to_columns: dict[int, list[str]] = {idx: [] for idx in range(len(table_names))}
    for table_idx, column_name in column_names:
        if table_idx != -1:
            table_to_columns.setdefault(table_idx, []).append(str(column_name))

    table_lines = []
    for table_idx, table_name in enumerate(table_names):
        columns = ", ".join(table_to_columns.get(table_idx, []))
        table_lines.append(f"- {table_name}({columns})")

    foreign_key_lines = []
    for source_idx, target_idx in foreign_keys:
        source_table_idx, source_column = column_names[source_idx]
        target_table_idx, target_column = column_names[target_idx]
        source_table = table_names[source_table_idx]
        target_table = table_names[target_table_idx]
        foreign_key_lines.append(f"- {source_table}.{source_column} -> {target_table}.{target_column}")

    schema_parts = ["Tables:", *table_lines]
    if foreign_key_lines:
        schema_parts.extend(["", "Foreign keys:", *foreign_key_lines])
    return "\n".join(schema_parts)


def build_schema_lookup(root: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for filename in ("tables.json", "test_tables.json"):
        path = root / filename
        if not path.exists():
            continue
        for entry in read_json(path):
            lookup[str(entry["db_id"])] = serialize_schema_entry(entry)
    return lookup


def load_spider_train_records(root: Path) -> list[dict[str, Any]]:
    samples = read_json(root / "train_spider.json")
    schema_lookup = build_schema_lookup(root)
    records = []
    for index, sample in enumerate(samples):
        db_id = str(sample["db_id"]).strip()
        records.append(
            {
                "id": index,
                "benchmark": "spider_data",
                "source_split": "train",
                "db_id": db_id,
                "question": str(sample["question"]).strip(),
                "gold_sql": str(sample["query"]).strip(),
                "schema": schema_lookup[db_id],
            }
        )
    return records


def load_synid_teacher_templates(prompt_dir: Path) -> tuple[str, str]:
    system_template = read_text(prompt_dir / "system_prompt.txt").replace("{json_schema}", JSON_SCHEMA)
    user_template = read_text(prompt_dir / "user_prompt.txt")
    return system_template, user_template


def build_teacher_user_prompt(template: str, record: dict[str, Any]) -> str:
    return template.format(
        question=record["question"],
        schema=record["schema"],
        gold_sql=record["gold_sql"],
    )
