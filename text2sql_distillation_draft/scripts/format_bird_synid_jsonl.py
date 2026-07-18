#!/usr/bin/env python3
"""Format the privileged SynID-SQL BIRD teacher training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from format_bird_jsonl import build_schema_lookup, write_jsonl


BIRD_ROOT = Path("benchmarks_2/bird")
TEACHER_SYSTEM_PROMPT = """You are a privileged SQL teacher for BIRD text-to-SQL tasks.

Your task is to generate a valid SQLite SQL query that answers the user's natural language question using the provided database schema, evidence, and reference solution.

Rules:
- Use only tables, columns, and relationships that exist in the provided schema.
- Use the evidence to resolve abbreviations, domain-specific terms, values, formulas, and implicit constraints.
- Do not invent schema elements or values that are not supported by the question, evidence, schema, or reference solution.
- Generate a single executable SQL query.
- Return only the columns needed to answer the question.
- Use DISTINCT when duplicates are possible and the question implies unique results.
- If the question requires aggregation, sorting, grouping, filtering, or limiting, use the correct SQL clauses.
- Use JOINs only when needed, and use schema-consistent join keys.
- Qualify column names when they may be ambiguous.
- Prefer concise, conventional SQLite SQL.
- Ensure the query is syntactically valid.

OUTPUT FORMAT:
{
  "sql": "The complete SQL query"
}

Return only the requested JSON object and nothing else."""

TEACHER_USER_TEMPLATE = """Problem: Given the following database schema and evidence, generate a SQL query to answer the question.

Database Schema:
{schema}

Question:
{question}

Evidence:
{evidence}

Here is a reference solution:
{gold_sql}

After understanding the reference solution, please try to solve this problem using your own approach.
Answer:"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_teacher_record(sample: dict[str, Any], schema_lookup: dict[str, str]) -> dict[str, str]:
    db_id = str(sample["db_id"]).strip()
    question = str(sample["question"]).strip()
    evidence = str(sample.get("evidence") or "").strip()
    gold_sql = str(sample["SQL"]).strip()
    schema = schema_lookup[db_id]

    return {
        "t_system_prompt": TEACHER_SYSTEM_PROMPT,
        "t_user_prompt": TEACHER_USER_TEMPLATE.format(
            schema=schema,
            question=question,
            evidence=evidence,
            gold_sql=gold_sql,
        ),
        "response": json.dumps({"sql": gold_sql}, ensure_ascii=False),
    }


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
            raise ValueError(f"Student and teacher responses differ at train row {index}.")


def format_teacher_train(root: Path, output_path: Path, student_train_path: Path) -> Path:
    samples = read_json(root / "train" / "train.json")
    schema_lookup = build_schema_lookup(root / "train" / "train_tables.json")
    rows = [make_teacher_record(sample, schema_lookup) for sample in samples]
    validate_student_alignment(rows, student_train_path)
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} privileged teacher rows to {output_path}")
    if student_train_path.exists():
        print(f"Validated response alignment with {student_train_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BIRD_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--student-train", type=Path)
    args = parser.parse_args()

    output_path = args.output or args.root / "format_data" / "teacher_train.jsonl"
    student_train_path = args.student_train or args.root / "format_data" / "train.jsonl"
    format_teacher_train(args.root, output_path, student_train_path)


if __name__ == "__main__":
    main()
