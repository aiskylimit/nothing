from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|<=|>=|<>|!=|[(),.*=<>+-/]")


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split())


def sql_tokens(sql: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(normalize_sql(sql))]
