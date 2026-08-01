#!/usr/bin/env python3
"""Build privileged teacher_train.jsonl from SynID augmentation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("benchmarks_2/synid_aug_v2/final_merged.jsonl")
DEFAULT_OUTPUT = Path("processed_data/benchmarks/spider_data/synid_privileged/qwen/teacher_train.jsonl")
DEFAULT_TRAIN_OUTPUT = Path("processed_data/benchmarks/spider_data/synid_privileged/qwen/train.jsonl")
DEFAULT_PROMPT_DIR = Path("prompts/single_turn/synid_teacher")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def response_sql(row: dict[str, Any], source: str) -> str:
    if source == "aug":
        sql = row.get("aug_sql") or row.get("candidate_sql")
        missing = "aug_sql/candidate_sql"
    elif source == "candidate":
        sql = row.get("candidate_sql")
        missing = "candidate_sql"
    elif source == "gold":
        sql = row.get("gold_sql")
        missing = "gold_sql"
    else:
        raise ValueError(f"Unsupported response source: {source}")

    if not sql:
        raise ValueError(f"Missing {missing} for row id={row.get('id')}")
    return str(sql).strip()


def build_row(
    row: dict[str, Any],
    *,
    system_prompt: str,
    user_prompt_template: str,
    response_source: str,
) -> dict[str, str]:
    for key in ("question", "schema", "gold_sql"):
        if not row.get(key):
            raise ValueError(f"Missing {key} for row id={row.get('id')}")

    return {
        "t_system_prompt": system_prompt,
        "t_user_prompt": user_prompt_template.format(
            schema=str(row["schema"]).strip(),
            question=str(row["question"]).strip(),
            gold_sql=str(row["gold_sql"]).strip(),
        ),
        "response": json.dumps({"sql": response_sql(row, response_source)}, ensure_ascii=False),
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_student_train_responses(
    train_path: Path,
    source_rows: list[dict[str, Any]],
    *,
    response_source: str,
) -> None:
    if not train_path.exists():
        raise FileNotFoundError(f"Student train file does not exist: {train_path}")

    student_rows = read_jsonl(train_path)
    if len(student_rows) != len(source_rows):
        raise ValueError(
            "Student train and teacher train row counts differ: "
            f"{len(student_rows)} != {len(source_rows)}. "
            f"Student file: {train_path}"
        )

    updated_rows = []
    for student, source_row in zip(student_rows, source_rows):
        next_student = dict(student)
        next_student["response"] = json.dumps(
            {"sql": response_sql(source_row, response_source)},
            ensure_ascii=False,
        )
        updated_rows.append(next_student)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in updated_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument(
        "--teacher-response-source",
        choices=["aug", "candidate", "gold"],
        default="aug",
        help="SQL source used for teacher_train.jsonl response.",
    )
    parser.add_argument(
        "--train-response-source",
        choices=["aug", "candidate", "gold"],
        default="aug",
        help="SQL source used for train.jsonl response.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    system_prompt = read_text(args.prompt_dir / "system_prompt.txt")
    user_prompt_template = read_text(args.prompt_dir / "user_prompt.txt")
    teacher_rows = [
        build_row(
            row,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            response_source=args.teacher_response_source,
        )
        for row in rows
    ]
    write_jsonl(args.output, teacher_rows)
    print(f"Wrote {len(teacher_rows)} rows to {args.output}")
    update_student_train_responses(
        args.train_output,
        rows,
        response_source=args.train_response_source,
    )
    print(f"Updated {len(teacher_rows)} student train responses in {args.train_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
