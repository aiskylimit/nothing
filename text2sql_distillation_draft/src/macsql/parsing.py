from __future__ import annotations

import json
import re
from typing import Any


SQL_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    payload = match.group(0)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def extract_sql(text: str) -> str:
    if not text:
        return ""

    parsed = parse_json_object(text)
    if parsed:
        for key in ("sql", "final_sql", "correct_sql", "query"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return clean_sql(value)

    blocks = SQL_BLOCK_RE.findall(text)
    if blocks:
        return clean_sql(blocks[-1])

    marker_patterns = (
        r"(?:final\s+sql|correct\s+sql|sql)\s*:\s*(.*)",
        r"(select\s+.*)",
        r"(with\s+.*)",
    )
    for pattern in marker_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return clean_sql(match.group(1))
    return clean_sql(text)


def clean_sql(sql: str) -> str:
    text = str(sql).strip()
    text = text.replace("\u2018", "`").replace("\u2019", "`")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"^```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    if text.endswith(";"):
        text = text[:-1].strip()
    return text


def extract_selector_schema(text: str) -> dict[str, Any]:
    parsed = parse_json_object(text)
    if not parsed:
        return {}
    schema = parsed.get("schema") or parsed.get("selected_schema") or parsed
    if not isinstance(schema, dict):
        return {}

    normalized: dict[str, Any] = {}
    for table, decision in schema.items():
        if not isinstance(table, str):
            continue
        if isinstance(decision, str):
            normalized[table] = decision
        elif isinstance(decision, list):
            normalized[table] = [str(column) for column in decision if str(column).strip()]
    return normalized

