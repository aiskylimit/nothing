from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .agents import DecomposerAgent, RefinerAgent, SelectorAgent
from .schema import MacSqlRecord, build_schema_strings, estimate_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_DIR = PROJECT_ROOT / "src" / "evaluator"
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from exec_eval import exec_on_db, postprocess  # noqa: E402


@dataclass
class MacSqlConfig:
    selector_threshold_tokens: int = 3500
    max_refine_rounds: int = 3
    execution_timeout: float = 30.0
    refine_empty_result: bool = True
    value_examples: int = 5


class MacSqlPipeline:
    def __init__(
        self,
        *,
        selector: SelectorAgent | None,
        decomposer: DecomposerAgent,
        refiner: RefinerAgent | None,
        config: MacSqlConfig,
    ):
        self.selector = selector
        self.decomposer = decomposer
        self.refiner = refiner
        self.config = config

    def run_record(self, record: MacSqlRecord) -> dict[str, Any]:
        row = record.row
        selected_schema: dict[str, Any] | None = None
        selector_raw = ""
        schema_text = record.full_schema
        fk_text = ""
        schema_for_selector = record.full_schema

        if self.selector and estimate_tokens(schema_for_selector) > self.config.selector_threshold_tokens:
            selected_schema, selector_raw = self.selector.select(
                db_id=row["db_id"],
                question=row["question"],
                schema=schema_for_selector,
            )
            schema_text, fk_text = build_schema_strings(
                record.schema_entry,
                db_path=record.db_path,
                selected_schema=selected_schema,
                value_examples=self.config.value_examples,
            )
        else:
            schema_text, fk_text = build_schema_strings(
                record.schema_entry,
                db_path=record.db_path,
                selected_schema=None,
                value_examples=self.config.value_examples,
            )

        pred_sql_initial, decomposition, decomposer_raw = self.decomposer.decompose(
            db_id=row["db_id"],
            question=row["question"],
            schema=schema_text,
            fk_str=fk_text,
        )
        pred_sql = pred_sql_initial
        refine_rounds: list[dict[str, Any]] = []
        execution = self._execute(record, pred_sql)

        if self.refiner is not None:
            for round_index in range(1, self.config.max_refine_rounds + 1):
                should_refine = not execution["ok"] or (
                    self.config.refine_empty_result and execution["ok"] and execution.get("row_count") == 0
                )
                if not should_refine:
                    break
                old_sql = pred_sql
                sqlite_error = execution.get("error") or "SQL executed but returned an empty result."
                exception_class = execution.get("exception_class") or "EmptyResult"
                pred_sql, refiner_raw = self.refiner.refine(
                    db_id=row["db_id"],
                    question=row["question"],
                    schema=schema_text,
                    fk_str=fk_text,
                    sql=old_sql,
                    sqlite_error=sqlite_error,
                    exception_class=exception_class,
                )
                execution = self._execute(record, pred_sql)
                refine_rounds.append(
                    {
                        "round": round_index,
                        "old_sql": old_sql,
                        "new_sql": pred_sql,
                        "sqlite_error": sqlite_error,
                        "exception_class": exception_class,
                        "raw_response": refiner_raw,
                        "execution_after": execution,
                    }
                )

        success = bool(pred_sql) and bool(execution.get("ok"))
        return {
            **row,
            "schema": schema_text,
            "selected_schema": selected_schema,
            "selector_raw_response": selector_raw,
            "decomposition": decomposition,
            "decomposer_raw_response": decomposer_raw,
            "pred_sql_initial": pred_sql_initial,
            "refine_rounds": refine_rounds,
            "pred_sql": pred_sql,
            "execution": execution,
            "success": success,
            "error": None if success else execution.get("error") or "empty final SQL",
        }

    def _execute(self, record: MacSqlRecord, sql: str) -> dict[str, Any]:
        if not sql:
            return {"ok": False, "error": "empty SQL", "exception_class": "ValueError", "row_count": None}
        if record.db_path is None:
            return {
                "ok": False,
                "error": f"SQLite database not found for db_id={record.row['db_id']}",
                "exception_class": "FileNotFoundError",
                "row_count": None,
            }
        db_paths = sorted(path for path in record.db_path.parent.iterdir() if ".sqlite" in path.name)
        if not db_paths:
            db_paths = [record.db_path]
        query = postprocess(sql)
        timeout = max(1, int(self.config.execution_timeout))
        row_count = None
        try:
            for db_path in db_paths:
                flag, payload = asyncio.run(
                    exec_on_db(
                        str(db_path),
                        query,
                        timeout=timeout,
                    )
                )
                if flag == "exception":
                    exception_class = payload.__name__ if isinstance(payload, type) else payload.__class__.__name__
                    return {
                        "ok": False,
                        "error": f"{db_path.name}: {payload}",
                        "exception_class": exception_class,
                        "row_count": None,
                        "checked_db_count": len(db_paths),
                        "executor": "spider_exec_eval.exec_on_db",
                    }
                if row_count is None:
                    row_count = len(payload)
        except (TimeoutError, OSError, ValueError) as exc:
            return {
                "ok": False,
                "error": str(exc),
                "exception_class": exc.__class__.__name__,
                "row_count": None,
                "checked_db_count": len(db_paths),
                "executor": "spider_exec_eval.exec_on_db",
            }
        return {
            "ok": True,
            "error": None,
            "exception_class": None,
            "row_count": row_count or 0,
            "checked_db_count": len(db_paths),
            "executor": "spider_exec_eval.exec_on_db",
        }


def run_pipeline(
    *,
    pipeline: MacSqlPipeline,
    records: list[MacSqlRecord],
    output_path: Path,
    flush_every: int | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(records, desc="Running MAC-SQL"), start=1):
        try:
            result = pipeline.run_record(record)
        except Exception as exc:
            result = {
                **record.row,
                "pred_sql_initial": "",
                "refine_rounds": [],
                "pred_sql": None,
                "execution": {"ok": False, "error": str(exc), "exception_class": exc.__class__.__name__},
                "success": False,
                "error": str(exc),
            }
        results.append(result)
        if flush_every and index % flush_every == 0:
            write_results(results, output_path)
    write_results(results, output_path)
    return results


def write_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, output_path)

