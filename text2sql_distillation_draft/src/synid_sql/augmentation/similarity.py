from __future__ import annotations

from difflib import SequenceMatcher

from .sql_normalize import normalize_sql


def sql_sequence_similarity(left_sql: str, right_sql: str) -> float:
    left = normalize_sql(left_sql).casefold()
    right = normalize_sql(right_sql).casefold()
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()
