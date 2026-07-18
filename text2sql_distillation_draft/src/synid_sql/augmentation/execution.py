from __future__ import annotations

import math
import sqlite3
import time
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_EVALUATOR_DIR = _ROOT / "src" / "evaluator"
if str(_EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALUATOR_DIR))


class SqlExecutionError(RuntimeError):
    pass


def resolve_sqlite_path(db_root: Path, db_id: str) -> Path:
    candidates = [
        db_root / db_id / f"{db_id}.sqlite",
        db_root / db_id / f"{db_id}.db",
        db_root / f"{db_id}.sqlite",
        db_root / f"{db_id}.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find SQLite database for db_id={db_id!r} under {db_root}")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return round(value, 6)
    return value


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    normalized = [tuple(_normalize_value(value) for value in row) for row in rows]
    return sorted(normalized, key=lambda row: repr(row))


def execute_sql(db_path: Path, sql: str, timeout_s: float = 30.0) -> list[tuple[Any, ...]]:
    deadline = time.monotonic() + timeout_s
    try:
        conn = sqlite3.connect(str(db_path))

        def progress_handler() -> int:
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(progress_handler, 1000)
        try:
            rows = conn.execute(sql).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise SqlExecutionError(str(exc)) from exc

    if time.monotonic() > deadline:
        raise TimeoutError(f"SQL execution exceeded {timeout_s} seconds")
    return _normalize_rows(rows)


def spider_exec_match(
    *,
    db_path: Path,
    gold_sql: str,
    candidate_sql: str,
    timeout_s: float = 60.0,
    plug_value: bool = False,
    keep_distinct: bool = False,
) -> bool:
    """Return Spider evaluator execution match for a gold/candidate pair."""
    try:
        from exec_eval import eval_exec_match
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing Spider evaluator dependency: {exc.name}. "
            "Install project requirements before running augmentation filtering."
        ) from exc

    score = eval_exec_match(
        db=str(db_path),
        p_str=candidate_sql,
        g_str=gold_sql,
        plug_value=plug_value,
        keep_distinct=keep_distinct,
        progress_bar_for_each_datapoint=False,
        timeout=int(timeout_s),
    )
    return bool(score)


def bird_exec_match(
    *,
    db_path: Path,
    gold_sql: str,
    candidate_sql: str,
    timeout_s: float = 60.0,
) -> bool:
    """Return BIRD evaluator execution match for a gold/candidate pair."""
    try:
        from bird_evaluation import execute_sql_pair
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing BIRD evaluator dependency: {exc.name}. "
            "Install project requirements before running augmentation filtering."
        ) from exc

    score, _error = execute_sql_pair(
        predicted_sql=candidate_sql,
        gold_sql=gold_sql,
        db_path=str(db_path),
        timeout=timeout_s,
    )
    return bool(score)
