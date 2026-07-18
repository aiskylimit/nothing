#!/usr/bin/env python3
"""Format Spider inference JSON files for the Spider evaluator.

Input files are the `*_sql_result.json` files emitted by `infer.py`.
For each input JSON, this script writes:

- `<name>.pred.sql`: one predicted SQL per line.
- `<name>.gold.sql`: one `gold_sql<TAB>db_id` row per line.
- `<name>.meta.jsonl`: row metadata for debugging/alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("infer_results/infer/qwen/spider_data")
DEFAULT_INPUT_GLOB = "*_sql_result.json"
EMPTY_PRED_SENTINEL = "__EMPTY_PRED__"
FIELD_ALIASES = {
    "sample_id": ("sample_id", "id", "question_id"),
    "db_id": ("db_id", "database_id", "db"),
    "gold_sql": ("gold_sql", "query", "gold", "target_sql"),
    "pred_sql": ("pred_sql", "prediction", "pred", "sql"),
}
BENCHMARK_SPLIT_FILES = {
    ("spider_data", "train"): Path("benchmarks_2/spider_data/train_spider.json"),
    ("spider_data", "dev"): Path("benchmarks_2/spider_data/dev.json"),
    ("spider_data", "test"): Path("benchmarks_2/spider_data/test.json"),
    ("spider_dk", "test"): Path("benchmarks_2/spider_dk/test.json"),
    ("spider_realistic", "test"): Path("benchmarks_2/spider_realistic/test.json"),
    ("spider_syn", "train"): Path("benchmarks_2/spider_syn/train_spider.json"),
    ("spider_syn", "test"): Path("benchmarks_2/spider_syn/test.json"),
    ("bird", "dev"): Path("benchmarks_2/bird/dev/dev.json"),
}


def one_line_sql(value: Any, *, empty_value: str = EMPTY_PRED_SENTINEL) -> str:
    if value is None:
        return empty_value
    text = str(value).strip()
    if not text:
        return empty_value
    return " ".join(text.split())


def one_line_pred_sql(value: Any) -> str:
    text = one_line_sql(value)
    if text == EMPTY_PRED_SENTINEL:
        return text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict) and parsed.get("sql"):
        return one_line_sql(parsed["sql"])
    return text


def get_field(row: dict[str, Any], field_name: str, *, required: bool = True) -> Any:
    for alias in FIELD_ALIASES[field_name]:
        if alias in row:
            return row[alias]
    if required:
        aliases = ", ".join(FIELD_ALIASES[field_name])
        raise KeyError(f"missing {field_name}; expected one of: {aliases}")
    return None


def parse_sample_index(sample_id: Any) -> int | None:
    if sample_id is None:
        return None
    text = str(sample_id)
    try:
        return int(text.rsplit(":", 1)[-1])
    except ValueError:
        return None


def load_gold_rows(input_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    benchmark = next((row.get("benchmark") for row in rows if row.get("benchmark")), None)
    split = next((row.get("split") for row in rows if row.get("split")), None)
    source_path = BENCHMARK_SPLIT_FILES.get((benchmark, split))
    if source_path is None or not source_path.exists():
        return None

    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError(f"{source_path} must contain a JSON list")
    print(f"refreshing gold_sql for {input_path.name} from {source_path}")
    return source_rows


def get_gold_sql(row: dict[str, Any], gold_rows: list[dict[str, Any]] | None) -> str:
    if gold_rows is not None:
        sample_id = get_field(row, "sample_id", required=False)
        index = parse_sample_index(sample_id)
        if index is not None and 0 <= index < len(gold_rows):
            query = gold_rows[index].get("query") or gold_rows[index].get("SQL")
            if query:
                return one_line_sql(query, empty_value="")
    return one_line_sql(get_field(row, "gold_sql"), empty_value="")


def output_prefix(input_path: Path) -> str:
    name = input_path.name
    for suffix in ("__full_sql_result.json", "_sql_result.json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    name = input_path.stem
    return name


def format_file(input_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{input_path} must contain a JSON list")

    prefix = output_prefix(input_path)
    pred_path = output_dir / f"{prefix}.pred.sql"
    gold_path = output_dir / f"{prefix}.gold.sql"
    meta_path = output_dir / f"{prefix}.meta.jsonl"

    pred_lines: list[str] = []
    gold_lines: list[str] = []
    meta_lines: list[str] = []
    empty_pred_count = 0
    gold_rows = load_gold_rows(input_path, rows)

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{input_path}: row {index} is not an object")

        try:
            pred_sql = one_line_pred_sql(get_field(row, "pred_sql"))
            gold_sql = get_gold_sql(row, gold_rows)
            db_id = str(get_field(row, "db_id")).strip()
            sample_id = get_field(row, "sample_id", required=False)
        except KeyError as exc:
            raise ValueError(f"{input_path}: row {index} {exc}") from exc
        if not gold_sql:
            raise ValueError(f"{input_path}: row {index} has empty gold_sql")
        if not db_id:
            raise ValueError(f"{input_path}: row {index} has empty db_id")
        if pred_sql == EMPTY_PRED_SENTINEL:
            empty_pred_count += 1

        pred_lines.append(pred_sql)
        gold_lines.append(f"{gold_sql}\t{db_id}")
        meta_lines.append(
            json.dumps(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "benchmark": row.get("benchmark"),
                    "split": row.get("split"),
                    "db_id": db_id,
                    "question": row.get("question"),
                    "success": row.get("success"),
                    "error": row.get("error"),
                },
                ensure_ascii=False,
            )
        )

    pred_path.write_text("\n".join(pred_lines) + "\n", encoding="utf-8")
    gold_path.write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    return {
        "input": str(input_path),
        "rows": len(rows),
        "empty_pred": empty_pred_count,
        "pred": str(pred_path),
        "gold": str(gold_path),
        "meta": str(meta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing inference JSON result files.",
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help=f"Glob for input files inside --input-dir. Defaults to {DEFAULT_INPUT_GLOB!r}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-dir>/formatted_data.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "formatted_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob(args.input_glob))
    if not input_files:
        raise FileNotFoundError(f"No files matching {args.input_glob!r} found in {input_dir}")

    summaries = [format_file(path, output_dir) for path in input_files]
    summary_path = output_dir / "format_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for item in summaries:
        print(
            f"formatted {Path(item['input']).name}: "
            f"rows={item['rows']} empty_pred={item['empty_pred']} "
            f"pred={item['pred']} gold={item['gold']}"
        )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
