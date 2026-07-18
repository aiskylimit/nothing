#!/usr/bin/env python3
"""Build SynID-SQL augmented train/teacher JSONL from accepted candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.format_bird_jsonl import make_record as make_bird_student_record
from scripts.format_spider_jsonl import make_record as make_spider_student_record
from src.synid_sql.augmentation.bird_records import (
    build_bird_schema_lookup,
    build_bird_teacher_user_prompt,
    load_bird_teacher_templates,
)
from src.synid_sql.augmentation.io import read_json, read_jsonl, read_text, write_json, write_jsonl
from src.synid_sql.augmentation.spider_records import (
    build_schema_lookup,
    build_teacher_user_prompt,
    load_synid_teacher_templates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["spider", "bird"], default="spider")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--accepted", type=Path, default=None)
    parser.add_argument("--teacher-prompt-dir", type=Path, default=Path("prompts/single_turn/synid_teacher"))
    parser.add_argument("--bird-prompt-dir", type=Path, default=Path("prompts/single_turn/bird_generator"))
    parser.add_argument("--student-output", type=Path, default=None)
    parser.add_argument("--teacher-output", type=Path, default=None)
    parser.add_argument("--copy-dev-test", action="store_true", help="Copy dev/test JSONL from root/format_data into output dir.")
    args = parser.parse_args()
    if args.root is None:
        args.root = Path("benchmarks_2/bird") if args.benchmark == "bird" else Path("benchmarks_2/spider_data")
    output_root = args.root / "synid_aug"
    if args.accepted is None:
        args.accepted = output_root / "accepted_all.jsonl"
    if args.student_output is None:
        args.student_output = output_root / "train.jsonl"
    if args.teacher_output is None:
        args.teacher_output = output_root / "teacher_train.jsonl"
    return args


def _load_spider_samples_by_id(root: Path) -> dict[int, dict[str, Any]]:
    samples = read_json(root / "train_spider.json")
    return {index: sample for index, sample in enumerate(samples)}


def _load_bird_samples_by_id(root: Path) -> dict[int, dict[str, Any]]:
    samples = read_json(root / "train" / "train.json")
    return {index: sample for index, sample in enumerate(samples)}


def _copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def build_spider_aug_jsonl(args: argparse.Namespace) -> None:
    accepted = read_jsonl(args.accepted)
    samples_by_id = _load_spider_samples_by_id(args.root)
    schema_lookup = build_schema_lookup(args.root)
    teacher_system_prompt, teacher_user_template = load_synid_teacher_templates(args.teacher_prompt_dir)

    student_rows = []
    teacher_rows = []
    for row in accepted:
        sample = dict(samples_by_id[int(row["id"])])
        aug_sql = str(row["aug_sql"]).strip()
        sample["query"] = aug_sql
        student_rows.append(make_spider_student_record(sample, schema_lookup))

        gold_record = {
            "question": str(row["question"]).strip(),
            "schema": str(row["schema"]).strip(),
            "gold_sql": str(row["gold_sql"]).strip(),
        }
        teacher_rows.append(
            {
                "t_system_prompt": teacher_system_prompt,
                "t_user_prompt": build_teacher_user_prompt(teacher_user_template, gold_record),
                "response": json.dumps({"sql": aug_sql}, ensure_ascii=False),
            }
        )

    write_jsonl(args.student_output, student_rows)
    write_jsonl(args.teacher_output, teacher_rows)

    copied = {}
    if args.copy_dev_test:
        args.student_output.parent.mkdir(parents=True, exist_ok=True)
        for name in ("dev.jsonl", "test.jsonl"):
            copied[name] = _copy_if_exists(args.root / "format_data" / name, args.student_output.parent / name)

    write_json(
        args.student_output.parent / "build_summary.json",
        {
            "accepted": len(accepted),
            "student_output": str(args.student_output),
            "teacher_output": str(args.teacher_output),
            "copied": copied,
        },
    )
    print(f"Wrote {len(student_rows)} student rows to {args.student_output}")
    print(f"Wrote {len(teacher_rows)} teacher rows to {args.teacher_output}")


def build_bird_aug_jsonl(args: argparse.Namespace) -> None:
    accepted = read_jsonl(args.accepted)
    samples_by_id = _load_bird_samples_by_id(args.root)
    schema_lookup = build_bird_schema_lookup(args.root, split="train")
    student_system_prompt = read_text(args.bird_prompt_dir / "system_prompt.txt")
    student_user_template = read_text(args.bird_prompt_dir / "user_prompt.txt")
    teacher_system_prompt, teacher_user_template = load_bird_teacher_templates()

    student_rows = []
    teacher_rows = []
    for row in accepted:
        sample = dict(samples_by_id[int(row["id"])])
        aug_sql = str(row["aug_sql"]).strip()
        sample["SQL"] = aug_sql
        student_rows.append(make_bird_student_record(sample, schema_lookup, student_system_prompt, student_user_template))

        gold_record = {
            "question": str(row["question"]).strip(),
            "evidence": str(row.get("evidence") or "").strip(),
            "schema": str(row["schema"]).strip(),
            "gold_sql": str(row["gold_sql"]).strip(),
        }
        teacher_rows.append(
            {
                "t_system_prompt": teacher_system_prompt,
                "t_user_prompt": build_bird_teacher_user_prompt(teacher_user_template, gold_record),
                "response": json.dumps({"sql": aug_sql}, ensure_ascii=False),
            }
        )

    write_jsonl(args.student_output, student_rows)
    write_jsonl(args.teacher_output, teacher_rows)

    copied = {}
    if args.copy_dev_test:
        args.student_output.parent.mkdir(parents=True, exist_ok=True)
        for name in ("val.jsonl", "test.jsonl"):
            copied[name] = _copy_if_exists(args.root / "format_data" / name, args.student_output.parent / name)

    write_json(
        args.student_output.parent / "build_summary.json",
        {
            "benchmark": "bird",
            "accepted": len(accepted),
            "student_output": str(args.student_output),
            "teacher_output": str(args.teacher_output),
            "copied": copied,
        },
    )
    print(f"Wrote {len(student_rows)} student rows to {args.student_output}")
    print(f"Wrote {len(teacher_rows)} teacher rows to {args.teacher_output}")


def main() -> None:
    args = parse_args()
    if args.benchmark == "bird":
        build_bird_aug_jsonl(args)
    else:
        build_spider_aug_jsonl(args)


if __name__ == "__main__":
    main()
