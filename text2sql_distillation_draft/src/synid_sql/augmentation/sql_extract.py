from __future__ import annotations

import json
import re
from typing import Any


_FENCED_BLOCK_RE = re.compile(r"```(?:json|sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _json_loads_loose(text: str) -> Any:
    return json.loads(
        text.replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
    )


def extract_sql(text: str) -> str:
    """Extract a SQL string from a strict or mildly malformed teacher response."""
    if not text or not text.strip():
        return ""

    raw = text.strip()
    candidates = [raw]
    candidates.extend(match.group(1).strip() for match in _FENCED_BLOCK_RE.finditer(raw))

    json_match = _JSON_BLOCK_RE.search(raw)
    if json_match:
        candidates.append(json_match.group(0).strip())

    for candidate in candidates:
        try:
            parsed = _json_loads_loose(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            value = parsed.get("sql") or parsed.get("SQL") or parsed.get("query")
            if isinstance(value, str):
                return value.strip()

    # Last-resort support for SQL-only fenced blocks or bare SQL responses.
    for candidate in candidates:
        stripped = candidate.strip()
        if re.match(r"^(with|select)\b", stripped, flags=re.IGNORECASE):
            return stripped
    return ""
