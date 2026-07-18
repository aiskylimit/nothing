from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .execution import SqlExecutionError, bird_exec_match, execute_sql, resolve_sqlite_path, spider_exec_match
from .jaccard import sql_token_jaccard
from .sql_normalize import normalize_sql


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
        if benchmark == "spider":
            execution_matches = spider_exec_match(
                db_path=db_path,
                gold_sql=gold_sql,
                candidate_sql=candidate_sql,
                timeout_s=timeout_s,
            )
        elif benchmark == "bird":
            execution_matches = bird_exec_match(
                db_path=db_path,
                gold_sql=gold_sql,
                candidate_sql=candidate_sql,
                timeout_s=timeout_s,
            )
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
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

    jaccard = sql_token_jaccard(gold_sql, candidate_sql)
    if jaccard >= gamma:
        return _reject(
            candidate,
            reason="jaccard_too_high",
            failure_stage="diversity_filter",
            repairable=True,
            candidate_sql=candidate_sql,
            jaccard=jaccard,
            gamma=gamma,
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
            "jaccard": jaccard,
            "gold_result_hash": _result_hash(gold_rows),
            "candidate_result_hash": _result_hash(candidate_rows),
        },
    )
