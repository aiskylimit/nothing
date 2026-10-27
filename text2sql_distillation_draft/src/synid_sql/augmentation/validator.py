from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .execution import SqlExecutionError, execute_sql, resolve_sqlite_path, spider_exec_match
from .similarity import sql_sequence_similarity
from .sql_normalize import normalize_sql


@contextmanager
def _timed(timings: dict[str, float], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - start


def _result_hash(rows: list[tuple[Any, ...]]) -> str:
    return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()


def _reject(
    candidate: dict[str, Any],
    *,
    reason: str,
    failure_stage: str,
    repairable: bool,
    **extra: Any,
) -> tuple[bool, dict[str, Any]]:
    return (
        False,
        {
            **candidate,
            "reason": reason,
            "failure_stage": failure_stage,
            "repairable": repairable,
            **extra,
        },
    )


def validate_candidate(
    candidate: dict[str, Any],
    *,
    benchmark: str = "spider",
    db_root: Path,
    gamma: float,
    timeout_s: float,
) -> tuple[bool, dict[str, Any]]:
    """Validate one candidate and attach per-step wall-clock seconds as ``row["timings"]``."""
    timings: dict[str, float] = {}
    start = time.perf_counter()
    ok, row = _validate_candidate(
        candidate,
        benchmark=benchmark,
        db_root=db_root,
        gamma=gamma,
        timeout_s=timeout_s,
        timings=timings,
    )
    timings["validate_total_s"] = time.perf_counter() - start
    row["timings"] = timings
    return ok, row


def _validate_candidate(
    candidate: dict[str, Any],
    *,
    benchmark: str,
    db_root: Path,
    gamma: float,
    timeout_s: float,
    timings: dict[str, float],
) -> tuple[bool, dict[str, Any]]:
    candidate_sql = normalize_sql(str(candidate.get("candidate_sql") or ""))
    if not candidate_sql:
        return _reject(
            candidate,
            reason="empty_sql",
            failure_stage="extract_candidate_sql",
            repairable=True,
        )

    db_id = str(candidate["db_id"])
    gold_sql = normalize_sql(str(candidate["gold_sql"]))
    try:
        db_path = resolve_sqlite_path(db_root, db_id)
    except Exception as exc:
        return _reject(
            candidate,
            reason="db_not_found",
            failure_stage="resolve_database",
            repairable=False,
            candidate_sql=candidate_sql,
            error=str(exc),
        )

    try:
        with _timed(timings, "exec_gold_s"):
            gold_rows = execute_sql(db_path, gold_sql, timeout_s=timeout_s)
    except (SqlExecutionError, TimeoutError) as exc:
        return _reject(
            candidate,
            reason="gold_execution_error",
            failure_stage="execute_gold_sql",
            repairable=False,
            candidate_sql=candidate_sql,
            error=str(exc),
        )

    try:
        with _timed(timings, "exec_cand_s"):
            candidate_rows = execute_sql(db_path, candidate_sql, timeout_s=timeout_s)
    except (SqlExecutionError, TimeoutError) as exc:
        return _reject(
            candidate,
            reason="candidate_execution_error",
            failure_stage="execute_candidate_sql",
            repairable=True,
            candidate_sql=candidate_sql,
            error=str(exc),
        )

    try:
        if benchmark != "spider":
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        with _timed(timings, "exec_match_s"):
            execution_matches = spider_exec_match(
                db_path=db_path,
                gold_sql=gold_sql,
                candidate_sql=candidate_sql,
                timeout_s=timeout_s,
            )
    except Exception as exc:
        return _reject(
            candidate,
            reason="evaluator_error",
            failure_stage="official_evaluator",
            repairable=True,
            candidate_sql=candidate_sql,
            evaluator=benchmark,
            error=str(exc),
        )

    if not execution_matches:
        return _reject(
            candidate,
            reason="execution_mismatch",
            failure_stage="execution_equivalence",
            repairable=True,
            candidate_sql=candidate_sql,
            gold_result_hash=_result_hash(gold_rows),
            candidate_result_hash=_result_hash(candidate_rows),
            gold_row_count=len(gold_rows),
            candidate_row_count=len(candidate_rows),
        )

    with _timed(timings, "similarity_s"):
        similarity = sql_sequence_similarity(gold_sql, candidate_sql)
    if similarity >= gamma:
        return _reject(
            candidate,
            reason="sql_similarity_too_high",
            failure_stage="diversity_filter",
            repairable=True,
            candidate_sql=candidate_sql,
            similarity=similarity,
            similarity_threshold=gamma,
            gold_result_hash=_result_hash(gold_rows),
            candidate_result_hash=_result_hash(candidate_rows),
        )

    return (
        True,
        {
            **candidate,
            "aug_sql": candidate_sql,
            "candidate_sql": candidate_sql,
            "status": "accepted",
            "similarity": similarity,
            "similarity_threshold": gamma,
            "gold_result_hash": _result_hash(gold_rows),
            "candidate_result_hash": _result_hash(candidate_rows),
        },
    )
