#!/usr/bin/env python3
"""Format Spider root benchmark JSON files into prompt/response JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from format_spider_variant_jsonl import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from format_spider_variant_jsonl import build_schema_lookup, write_jsonl


SPIDER_ROOT = Path("benchmarks_2/spider_data")
SPLIT_CONFIG = {
    "train": ("train_spider.json", "tables.json"),
    "dev": ("dev.json", "tables.json"),
    "test": ("test.json", "test_tables.json"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_record(sample: dict[str, Any], schema_lookup: dict[str, str]) -> dict[str, str]:
    db_id = str(sample["db_id"]).strip()
    question = str(sample["question"]).strip()
    sql = str(sample["query"]).strip()
    schema = schema_lookup[db_id]

    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT_TEMPLATE.format(question=question, schema=schema),
        "response": json.dumps({"sql": sql}, ensure_ascii=False),
    }


def format_split(root: Path, split: str) -> Path:
    if split not in SPLIT_CONFIG:
        available = ", ".join(sorted(SPLIT_CONFIG))
        raise ValueError(f"Unsupported split {split!r}. Available splits: {available}")

    data_name, tables_name = SPLIT_CONFIG[split]
    samples = read_json(root / data_name)
    schema_lookup = build_schema_lookup(root / tables_name)
    rows = [make_record(sample, schema_lookup) for sample in samples]

    output_path = root / "format_data" / f"{split}.jsonl"
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SPIDER_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    for split in args.splits:
        format_split(args.root, split)


if __name__ == "__main__":
    main()
