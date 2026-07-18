from __future__ import annotations

from .sql_normalize import sql_tokens


def sql_token_jaccard(left_sql: str, right_sql: str) -> float:
    left = set(sql_tokens(left_sql))
    right = set(sql_tokens(right_sql))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
