#!/usr/bin/env python3
"""Format the privileged SynID-SQL Spider teacher training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPIDER_ROOT = Path("benchmarks_2/spider_data")
TEACHER_PROMPT_DIR = Path("prompts/single_turn/synid_teacher")
JSON_SCHEMA = json.dumps({"sql": "The complete SQL query"}, indent=2)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


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
        foreign_key_lines.append(
            f"- {source_table}.{source_column} -> {target_table}.{target_column}"
        )

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


def make_teacher_record(
    sample: dict[str, Any],
    schema_lookup: dict[str, str],
    *,
    teacher_system_template: str,
    teacher_user_template: str,
) -> dict[str, str]:
    db_id = str(sample["db_id"])
    question = str(sample["question"]).strip()
    gold_sql = str(sample["query"]).strip()
    schema = schema_lookup[db_id]

    return {
        "t_system_prompt": teacher_system_template.replace(
            "{json_schema}",
            JSON_SCHEMA,
        ),
        "t_user_prompt": teacher_user_template.format(
            question=question,
            schema=schema,
            gold_sql=gold_sql,
        ),
        "response": json.dumps({"sql": gold_sql}, ensure_ascii=False),
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_student_alignment(
    teacher_rows: list[dict[str, str]],
    student_train_path: Path,
) -> None:
    if not student_train_path.exists():
        return

    student_rows = read_jsonl(student_train_path)
    if len(student_rows) != len(teacher_rows):
        raise ValueError(
            "Student and teacher train row counts differ: "
            f"{len(student_rows)} != {len(teacher_rows)}."
        )
    for index, (student_row, teacher_row) in enumerate(zip(student_rows, teacher_rows)):
        if student_row.get("response") != teacher_row["response"]:
            raise ValueError(
                f"Student and teacher responses differ at train row {index}."
            )


def format_teacher_train(
    root: Path,
    output_path: Path,
    teacher_prompt_dir: Path,
    student_train_path: Path,
) -> Path:
    samples = read_json(root / "train_spider.json")
    schema_lookup = build_schema_lookup(root)
    teacher_system_template = read_text(teacher_prompt_dir / "system_prompt.txt")
    teacher_user_template = read_text(teacher_prompt_dir / "user_prompt.txt")

    rows = [
        make_teacher_record(
            sample,
            schema_lookup,
            teacher_system_template=teacher_system_template,
            teacher_user_template=teacher_user_template,
        )
        for sample in samples
    ]
    validate_student_alignment(rows, student_train_path)
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} privileged teacher rows to {output_path}")
    if student_train_path.exists():
        print(f"Validated response alignment with {student_train_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SPIDER_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--teacher-prompt-dir", type=Path, default=TEACHER_PROMPT_DIR)
    parser.add_argument("--student-train", type=Path)
    args = parser.parse_args()

    output_path = args.output or args.root / "format_data" / "teacher_train.jsonl"
    student_train_path = args.student_train or args.root / "format_data" / "train.jsonl"
    format_teacher_train(
        args.root,
        output_path,
        args.teacher_prompt_dir,
        student_train_path,
    )


if __name__ == "__main__":
    main()
