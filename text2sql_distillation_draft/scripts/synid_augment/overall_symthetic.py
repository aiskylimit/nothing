#!/usr/bin/env python3
"""Merge accepted SynID candidates with recovered final rejections.

This implements the final selection rule used by the reformulation pipeline:
after the retry budget, keep the execution-correct candidate with the lowest
normalized SQL SequenceMatcher similarity. If no execution-correct candidate
exists, fall back to the original gold SQL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SIMILARITY_TOO_HIGH_REASONS = {"sql_similarity_too_high", "jaccard_too_high"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_similarity(row: dict[str, Any]) -> float:
    if row.get("similarity") is not None:
        return float(row["similarity"])
    if row.get("jaccard") is not None:
        return float(row["jaccard"])
    return 1.0


def select_best_rejected_candidates(loops_dir: Path, final_rejected_ids: set[str]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for loop_file in sorted(loops_dir.glob("loop_*/rejected.jsonl")):
        for candidate in read_jsonl(loop_file):
            candidate_id = str(candidate.get("id"))
            if candidate_id not in final_rejected_ids:
                continue
            if candidate.get("reason") not in SIMILARITY_TOO_HIGH_REASONS:
                continue
            previous = best.get(candidate_id)
            if previous is None or row_similarity(candidate) < row_similarity(previous):
                best[candidate_id] = candidate
    return best


def recover_rows(
    final_rejected_rows: list[dict[str, Any]],
    best_candidates: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    recovered_rows = []
    stats = {"recovered_from_loops": 0, "fallback_to_gold": 0}

    for row in final_rejected_rows:
        row_id = str(row["id"])
        recovered = dict(row)
        best = best_candidates.get(row_id)
        if best is not None:
            candidate_sql = best.get("aug_sql") or best.get("candidate_sql")
            if not candidate_sql:
                raise ValueError(f"Missing candidate SQL for recovered row id={row_id}")
            recovered["candidate_sql"] = str(candidate_sql).strip()
            recovered["aug_sql"] = str(candidate_sql).strip()
            recovered["similarity"] = row_similarity(best)
            recovered["similarity_threshold"] = best.get("similarity_threshold") or best.get("gamma")
            recovered["recovery_source"] = f"loop_{best.get('loop', 'unknown')}"
            stats["recovered_from_loops"] += 1
        else:
            gold_sql = row.get("gold_sql") or row.get("query")
            if not gold_sql:
                raise ValueError(f"Missing gold_sql/query for fallback row id={row_id}")
            recovered["candidate_sql"] = str(gold_sql).strip()
            recovered["aug_sql"] = str(gold_sql).strip()
            recovered["similarity"] = 1.0
            recovered["recovery_source"] = "gold_fallback"
            stats["fallback_to_gold"] += 1
        recovered_rows.append(recovered)

    return recovered_rows, stats


def sort_by_id(rows: list[dict[str, Any]]) -> None:
    try:
        rows.sort(key=lambda row: int(row["id"]))
    except (KeyError, ValueError):
        rows.sort(key=lambda row: str(row.get("id", "")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("benchmarks_2/spider_data/synid_aug_v2_lora"))
    parser.add_argument("--accepted", type=Path, default=None)
    parser.add_argument("--rejected-final", type=Path, default=None)
    parser.add_argument("--loops-dir", type=Path, default=None)
    parser.add_argument("--recovered-output", type=Path, default=None)
    parser.add_argument("--merged-output", type=Path, default=None)
    args = parser.parse_args()

    accepted_path = args.accepted or args.base_dir / "accepted_all.jsonl"
    rejected_final_path = args.rejected_final or args.base_dir / "rejected_final.jsonl"
    loops_dir = args.loops_dir or args.base_dir / "loops"
    recovered_output = args.recovered_output or args.base_dir / "recovered_final.jsonl"
    merged_output = args.merged_output or args.base_dir / "final_merged.jsonl"

    if not rejected_final_path.exists():
        raise FileNotFoundError(f"Missing final rejected file: {rejected_final_path}")
    if not loops_dir.exists():
        raise FileNotFoundError(f"Missing loops directory: {loops_dir}")

    final_rejected_rows = read_jsonl(rejected_final_path)
    final_rejected_ids = {str(row["id"]) for row in final_rejected_rows}
    best_candidates = select_best_rejected_candidates(loops_dir, final_rejected_ids)
    recovered_rows, stats = recover_rows(final_rejected_rows, best_candidates)
    write_jsonl(recovered_output, recovered_rows)

    accepted_rows = read_jsonl(accepted_path) if accepted_path.exists() else []
    for row in accepted_rows:
        row.setdefault("recovery_source", "accepted")

    final_rows = [*accepted_rows, *recovered_rows]
    sort_by_id(final_rows)
    write_jsonl(merged_output, final_rows)

    print(f"accepted={len(accepted_rows)}")
    print(f"recovered={len(recovered_rows)}")
    print(f"recovered_from_loops={stats['recovered_from_loops']}")
    print(f"fallback_to_gold={stats['fallback_to_gold']}")
    print(f"merged={len(final_rows)}")
    print(f"recovered_output={recovered_output}")
    print(f"merged_output={merged_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
