from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infer import get_benchmark_paths, load_records, read_json
from src.synid_sql.augmentation.execution import resolve_sqlite_path


@dataclass
class MacSqlRecord:
    row: dict[str, Any]
    schema_entry: dict[str, Any]
    db_path: Path | None
    full_schema: str


DEFAULT_DB_ROOTS = {
    ("spider_data", "train"): Path("benchmarks/spider_data/database"),
    ("spider_data", "dev"): Path("benchmarks/spider_data/database"),
    ("spider_data", "test"): Path("benchmarks/spider_data/test_database"),
    "spider_syn": Path("benchmarks/spider_data/database"),
    "spider_realistic": Path("benchmarks/spider_data/database"),
    "spider_dk": Path("benchmarks/spider_dk/database"),
    "bird": Path("benchmarks/bird/dev/dev_databases"),
}


def load_table_entries(benchmark: str, split: str) -> dict[str, dict[str, Any]]:
    _, tables_path = get_benchmark_paths(benchmark, split)
    return {str(entry["db_id"]): entry for entry in read_json(tables_path)}


def default_db_root(benchmark: str, split: str) -> Path | None:
    keyed = DEFAULT_DB_ROOTS.get((benchmark, split))
    if keyed is not None:
        return keyed
    return DEFAULT_DB_ROOTS.get(benchmark)


def load_macsql_records(
    *,
    benchmark: str,
    split: str,
    db_filter: str | None,
    limit: int | None,
    db_root: Path | None,
    value_examples: int,
) -> list[MacSqlRecord]:
    rows = load_records(benchmark, split, db_filter, limit)
    entries = load_table_entries(benchmark, split)
    root = db_root or default_db_root(benchmark, split)
    output: list[MacSqlRecord] = []
    for row in rows:
        db_id = str(row["db_id"])
        db_path = None
        if root is not None:
            try:
                db_path = resolve_sqlite_path(root, db_id)
            except FileNotFoundError:
                db_path = None
        entry = entries[db_id]
        full_schema, _ = build_schema_strings(entry, db_path=db_path, selected_schema=None, value_examples=value_examples)
        output.append(MacSqlRecord(row=row, schema_entry=entry, db_path=db_path, full_schema=full_schema))
    return output


def build_schema_strings(
    entry: dict[str, Any],
    *,
    db_path: Path | None,
    selected_schema: dict[str, Any] | None,
    value_examples: int,
) -> tuple[str, str]:
    table_names = entry.get("table_names_original") or entry.get("table_names") or []
    column_names = entry.get("column_names_original") or entry.get("column_names") or []
    foreign_keys = entry.get("foreign_keys") or []
    primary_keys = set()
    for pk in entry.get("primary_keys") or []:
        if isinstance(pk, list):
            primary_keys.update(int(item) for item in pk)
        else:
            primary_keys.add(int(pk))

    fk_col_ids = set()
    for source_idx, target_idx in foreign_keys:
        fk_col_ids.add(source_idx)
        fk_col_ids.add(target_idx)

    table_to_column_ids: dict[int, list[int]] = {idx: [] for idx in range(len(table_names))}
    for col_idx, (table_idx, _) in enumerate(column_names):
        if table_idx >= 0:
            table_to_column_ids.setdefault(table_idx, []).append(col_idx)

    conn = None
    if db_path is not None and db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda raw: raw.decode(errors="ignore")

    try:
        table_blocks: list[str] = []
        for table_idx, table_name in enumerate(table_names):
            column_ids = table_to_column_ids.get(table_idx, [])
            kept_ids = _select_column_ids(
                table_name=str(table_name),
                column_ids=column_ids,
                column_names=column_names,
                selected_schema=selected_schema,
                key_ids=primary_keys | fk_col_ids,
            )
            if not kept_ids:
                continue
            lines = [f"# Table: {table_name}", "["]
            for col_idx in kept_ids:
                _, column_name = column_names[col_idx]
                pieces = [str(column_name)]
                if col_idx in primary_keys:
                    pieces.append("primary key")
                if col_idx in fk_col_ids:
                    pieces.append("foreign key")
                examples = _value_examples(conn, str(table_name), str(column_name), value_examples)
                if examples:
                    pieces.append(f"Value examples: {examples}")
                lines.append(f"({', '.join(pieces)}),")
            lines.append("]")
            table_blocks.append("\n".join(lines))
    finally:
        if conn is not None:
            conn.close()

    fk_lines: list[str] = []
    for src_idx, dst_idx in foreign_keys:
        src_table_idx, src_col = column_names[src_idx]
        dst_table_idx, dst_col = column_names[dst_idx]
        if src_table_idx < 0 or dst_table_idx < 0:
            continue
        src_table = table_names[src_table_idx]
        dst_table = table_names[dst_table_idx]
        if selected_schema and not (_table_kept(str(src_table), selected_schema) and _table_kept(str(dst_table), selected_schema)):
            continue
        fk_lines.append(f"{src_table}.`{src_col}` = {dst_table}.`{dst_col}`")

    schema_text = "\n".join(table_blocks)
    fk_text = "\n".join(fk_lines) if fk_lines else "None"
    return schema_text, fk_text


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _table_kept(table_name: str, selected_schema: dict[str, Any]) -> bool:
    decision = selected_schema.get(table_name)
    return decision is None or decision != "drop_all"


def _select_column_ids(
    *,
    table_name: str,
    column_ids: list[int],
    column_names: list[Any],
    selected_schema: dict[str, Any] | None,
    key_ids: set[int],
) -> list[int]:
    if not selected_schema:
        return column_ids

    decision = selected_schema.get(table_name)
    if decision == "drop_all":
        return column_ids[:6]
    if decision in (None, "", "keep_all"):
        return column_ids
    if not isinstance(decision, list):
        return column_ids

    wanted = {str(item).lower() for item in decision}
    kept = [
        col_idx
        for col_idx in column_ids
        if col_idx in key_ids or str(column_names[col_idx][1]).lower() in wanted
    ]
    if len(column_ids) > 6 and len(kept) < 6:
        for col_idx in column_ids:
            if col_idx not in kept:
                kept.append(col_idx)
            if len(kept) >= 6:
                break
    return kept


def _value_examples(conn: sqlite3.Connection | None, table: str, column: str, limit: int) -> str:
    if conn is None or limit <= 0:
        return ""
    lower = column.lower()
    if lower.endswith("id") or lower.endswith("url") or lower.endswith("email"):
        return ""
    try:
        rows = conn.execute(
            f"SELECT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL "
            f"GROUP BY `{column}` ORDER BY COUNT(*) DESC LIMIT {int(limit)}"
        ).fetchall()
    except sqlite3.Error:
        return ""
    values = [row[0] for row in rows if row and row[0] not in ("", None)]
    clean = []
    for value in values:
        text = str(value)
        if len(text) > 50 or "http://" in text or "https://" in text or "@" in text:
            continue
        clean.append(value)
    return str(clean[:limit]) if clean else ""
