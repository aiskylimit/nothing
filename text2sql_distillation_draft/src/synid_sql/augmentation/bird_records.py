from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import read_json
from .spider_records import serialize_schema_entry


JSON_SCHEMA = json.dumps({"sql": "The complete SQL query"}, indent=2)

BIRD_TEACHER_SYSTEM_PROMPT = """You are a privileged SQL teacher for BIRD text-to-SQL tasks.

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
{json_schema}

Return only the requested JSON object and nothing else."""

BIRD_TEACHER_USER_TEMPLATE = """Problem: Given the following database schema and evidence, generate a SQL query to answer the question.

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


def build_bird_schema_lookup(root: Path, split: str = "train") -> dict[str, str]:
    tables_path = root / split / f"{split}_tables.json"
    return {str(entry["db_id"]): serialize_schema_entry(entry) for entry in read_json(tables_path)}


def load_bird_train_records(root: Path) -> list[dict[str, Any]]:
    samples = read_json(root / "train" / "train.json")
    schema_lookup = build_bird_schema_lookup(root, split="train")
    records = []
    for index, sample in enumerate(samples):
        db_id = str(sample["db_id"]).strip()
        records.append(
            {
                "id": index,
                "benchmark": "bird",
                "source_split": "train",
                "db_id": db_id,
                "question": str(sample["question"]).strip(),
                "evidence": str(sample.get("evidence") or "").strip(),
                "gold_sql": str(sample["SQL"]).strip(),
                "schema": schema_lookup[db_id],
            }
        )
    return records


def load_bird_teacher_templates() -> tuple[str, str]:
    return BIRD_TEACHER_SYSTEM_PROMPT.format(json_schema=JSON_SCHEMA), BIRD_TEACHER_USER_TEMPLATE


def build_bird_teacher_user_prompt(template: str, record: dict[str, Any]) -> str:
    return template.format(
        question=record["question"],
        evidence=record.get("evidence") or "",
        schema=record["schema"],
        gold_sql=record["gold_sql"],
    )
