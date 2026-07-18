import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sqlite3
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


BIRD_SEPARATOR = "\t----- bird -----\t"
DIFFICULTY_LEVELS = ("simple", "moderate", "challenging")


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _clean_prediction_sql(sql: str) -> str:
    sql = sql.strip()
    if not sql:
        return ""
    if sql.startswith("{") and sql.endswith("}"):
        try:
            payload = json.loads(sql)
        except json.JSONDecodeError:
            match = re.match(r'^\{\s*"sql"\s*:\s*"(.*)"\s*\}\s*$', sql)
            if not match:
                return sql
            return match.group(1).replace(r"\'", "'").replace(r"\"", '"').strip()
        if isinstance(payload, dict):
            value = payload.get("sql")
            if isinstance(value, str):
                return value.strip()
    return sql


def _replace_multiple_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_gold_line(line: str) -> Tuple[str, str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        raise ValueError(f"Gold line must contain SQL and db_id separated by a tab: {line!r}")
    return "\t".join(parts[:-1]).strip(), parts[-1].strip()


def _parse_prediction_line(line: str) -> Tuple[str, Optional[str]]:
    text = line.rstrip("\n")
    if BIRD_SEPARATOR in text:
        sql, db_id = text.split(BIRD_SEPARATOR, 1)
        return _clean_prediction_sql(sql), db_id.strip()

    parts = text.split("\t")
    if len(parts) >= 2:
        return _clean_prediction_sql("\t".join(parts[:-1])), parts[-1].strip()

    return _clean_prediction_sql(text), None


def load_gold_sqls(gold_path: str) -> List[Tuple[str, str]]:
    with open(gold_path, encoding="utf-8") as file:
        return [_parse_gold_line(line) for line in file if line.strip()]


def load_pred_sqls(pred_path: str) -> List[Tuple[str, Optional[str]]]:
    if pred_path.endswith(".json"):
        payload = _read_json(pred_path)
        if isinstance(payload, dict):
            values = [
                payload[key]
                for key in sorted(
                    payload,
                    key=lambda item: (0, int(item)) if str(item).isdigit() else (1, str(item)),
                )
            ]
        elif isinstance(payload, list):
            values = payload
        else:
            raise ValueError(f"Unsupported prediction JSON shape in {pred_path}")

        predictions = []
        for value in values:
            if isinstance(value, str):
                predictions.append(_parse_prediction_line(value))
            elif isinstance(value, (list, tuple)):
                if len(value) >= 2 and isinstance(value[1], str):
                    predictions.append(_parse_prediction_line(value[1]))
                elif len(value) >= 1 and isinstance(value[0], str):
                    predictions.append(_parse_prediction_line(value[0]))
                else:
                    predictions.append(("", None))
            elif isinstance(value, dict):
                sql = value.get("sql") or value.get("pred_sql") or value.get("prediction") or ""
                db_id = value.get("db_id")
                predictions.append((_clean_prediction_sql(str(sql)), str(db_id) if db_id else None))
            else:
                predictions.append(("", None))
        return predictions

    with open(pred_path, encoding="utf-8") as file:
        return [_parse_prediction_line(line) for line in file if line.strip()]


def load_difficulties(diff_json_path: Optional[str], total: int) -> List[str]:
    if not diff_json_path:
        return ["total"] * total

    contents = _read_json(diff_json_path)
    difficulties = []
    for item in contents[:total]:
        difficulty = item.get("difficulty", "unknown") if isinstance(item, dict) else "unknown"
        difficulties.append(difficulty)

    if len(difficulties) < total:
        difficulties.extend(["unknown"] * (total - len(difficulties)))
    return difficulties


def database_path(db_root_path: str, db_id: str) -> str:
    return os.path.join(db_root_path, db_id, f"{db_id}.sqlite")


def _progress(items, enabled: bool, desc: str):
    return tqdm(items, desc=desc, unit="query") if enabled else items


def _run_tasks(worker, tasks: Sequence[Dict[str, Any]], num_cpus: int, show_progress: bool, desc: str):
    if num_cpus <= 0:
        raise ValueError("num_cpus must be positive.")
    if num_cpus == 1:
        return [worker(task) for task in _progress(tasks, show_progress, desc)]

    with mp.Pool(processes=num_cpus) as pool:
        iterator = pool.imap_unordered(worker, tasks)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc=desc, unit="query")
        results = list(iterator)
    return sorted(results, key=lambda result: result["sql_idx"])


def execute_sql_pair(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
    timeout: Optional[float],
) -> Tuple[int, Optional[str]]:
    if not predicted_sql.strip():
        return 0, "empty prediction"

    conn = sqlite3.connect(db_path)
    try:
        if timeout is not None and timeout > 0:
            deadline = time.monotonic() + timeout

            def stop_after_timeout() -> int:
                return int(time.monotonic() >= deadline)

            conn.set_progress_handler(stop_after_timeout, 1000)

        cursor = conn.cursor()
        try:
            cursor.execute(predicted_sql)
            predicted_result = cursor.fetchall()
            cursor.execute(gold_sql)
            gold_result = cursor.fetchall()
        finally:
            cursor.close()
    except Exception as exc:
        return 0, str(exc)
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()

    return int(set(predicted_result) == set(gold_result)), None


def _execute_bird_execution_task(task: Dict[str, Any]) -> Dict[str, Any]:
    if task["db_id"] != task["gold_db_id"]:
        return {
            "sql_idx": task["sql_idx"],
            "db_id": task["db_id"],
            "difficulty": task["difficulty"],
            "res": 0,
            "error": f"predicted db_id {task['db_id']!r} does not match gold db_id {task['gold_db_id']!r}",
        }

    score, error = execute_sql_pair(task["pred_sql"], task["gold_sql"], task["db_path"], task["timeout"])
    return {
        "sql_idx": task["sql_idx"],
        "db_id": task["gold_db_id"],
        "difficulty": task["difficulty"],
        "res": score,
        "error": error,
    }


def _execute_sql_timed(sql: str, db_path: str, timeout: Optional[float]) -> float:
    conn = sqlite3.connect(db_path)
    try:
        if timeout is not None and timeout > 0:
            deadline = time.monotonic() + timeout

            def stop_after_timeout() -> int:
                return int(time.monotonic() >= deadline)

            conn.set_progress_handler(stop_after_timeout, 1000)

        cursor = conn.cursor()
        try:
            start = time.perf_counter()
            cursor.execute(sql)
            return time.perf_counter() - start
        finally:
            cursor.close()
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


def _filter_abnormal(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    filtered = [value for value in values if mean - 3 * std < value < mean + 3 * std]
    return filtered or list(values)


def compute_ves_time_ratio(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
    iterate_num: int,
    timeout: Optional[float],
) -> Tuple[float, Optional[str]]:
    score, error = execute_sql_pair(predicted_sql, gold_sql, db_path, timeout)
    if score == 0:
        return 0.0, error

    ratios = []
    total_timeout = timeout * iterate_num if timeout is not None and timeout > 0 else None
    deadline = time.monotonic() + total_timeout if total_timeout is not None else None
    for _ in range(iterate_num):
        remaining = None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 0.0, "timeout"

        try:
            predicted_time = _execute_sql_timed(predicted_sql, db_path, remaining)
            gold_time = _execute_sql_timed(gold_sql, db_path, remaining)
        except Exception as exc:
            return 0.0, str(exc)
        if predicted_time > 0:
            ratios.append(gold_time / predicted_time)

    processed_ratios = _filter_abnormal(ratios)
    if not processed_ratios:
        return 0.0, "empty timing ratios"
    return sum(processed_ratios) / len(processed_ratios), None


def _execute_bird_ves_task(task: Dict[str, Any]) -> Dict[str, Any]:
    if task["db_id"] != task["gold_db_id"]:
        return {
            "sql_idx": task["sql_idx"],
            "db_id": task["db_id"],
            "difficulty": task["difficulty"],
            "time_ratio": 0.0,
            "ves": 0.0,
            "error": f"predicted db_id {task['db_id']!r} does not match gold db_id {task['gold_db_id']!r}",
        }

    time_ratio, error = compute_ves_time_ratio(
        task["pred_sql"],
        task["gold_sql"],
        task["db_path"],
        task["iterate_num"],
        task["timeout"],
    )
    return {
        "sql_idx": task["sql_idx"],
        "db_id": task["gold_db_id"],
        "difficulty": task["difficulty"],
        "time_ratio": time_ratio,
        "ves": math.sqrt(time_ratio) * 100 if time_ratio > 0 else 0.0,
        "error": error,
    }


def _build_bird_tasks(
    pred_path: str,
    gold_path: str,
    db_root_path: str,
    diff_json_path: Optional[str],
    timeout: Optional[float],
    iterate_num: Optional[int] = None,
) -> List[Dict[str, Any]]:
    predictions = load_pred_sqls(pred_path)
    golds = load_gold_sqls(gold_path)

    if len(predictions) != len(golds):
        raise ValueError(f"Prediction count does not match gold: {len(predictions)} != {len(golds)}")

    difficulties = load_difficulties(diff_json_path, len(golds))
    tasks = []
    for index, ((pred_sql, pred_db_id), (gold_sql, gold_db_id)) in enumerate(zip(predictions, golds)):
        db_id = pred_db_id or gold_db_id
        db_path = database_path(db_root_path, gold_db_id)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file does not exist: {db_path}")

        task = {
            "sql_idx": index,
            "pred_sql": pred_sql,
            "pred_db_id": pred_db_id,
            "gold_sql": gold_sql,
            "gold_db_id": gold_db_id,
            "db_id": db_id,
            "db_path": db_path,
            "difficulty": difficulties[index],
            "timeout": timeout,
        }
        if iterate_num is not None:
            task["iterate_num"] = iterate_num
        tasks.append(task)
    return tasks


def evaluate_bird_execution(
    pred_path: str,
    gold_path: str,
    db_root_path: str,
    diff_json_path: Optional[str] = None,
    timeout: Optional[float] = 30.0,
    show_progress: bool = False,
    num_cpus: int = 1,
) -> List[Dict[str, Any]]:
    tasks = _build_bird_tasks(pred_path, gold_path, db_root_path, diff_json_path, timeout)
    return _run_tasks(_execute_bird_execution_task, tasks, num_cpus, show_progress, "BIRD EX")


def evaluate_bird_ves(
    pred_path: str,
    gold_path: str,
    db_root_path: str,
    diff_json_path: Optional[str] = None,
    iterate_num: int = 100,
    timeout: Optional[float] = 30.0,
    show_progress: bool = False,
    num_cpus: int = 1,
) -> List[Dict[str, Any]]:
    if iterate_num <= 0:
        raise ValueError("iterate_num must be positive for BIRD VES evaluation.")

    tasks = _build_bird_tasks(
        pred_path,
        gold_path,
        db_root_path,
        diff_json_path,
        timeout,
        iterate_num=iterate_num,
    )
    return _run_tasks(_execute_bird_ves_task, tasks, num_cpus, show_progress, "BIRD VES")


def save_bird_execution_details(
    results: Sequence[Dict[str, Any]],
    pred_path: str,
    gold_path: str,
    diff_json_path: Optional[str],
    output_path: str,
) -> None:
    predictions = load_pred_sqls(pred_path)
    golds = load_gold_sqls(gold_path)
    diff_items = _read_json(diff_json_path) if diff_json_path else [{} for _ in results]
    by_index = {int(result["sql_idx"]): result for result in results}

    details = []
    for index, ((pred_sql, _), (gold_sql, gold_db_id)) in enumerate(zip(predictions, golds)):
        source = diff_items[index] if index < len(diff_items) and isinstance(diff_items[index], dict) else {}
        item = dict(source)
        item["pred"] = _replace_multiple_spaces(pred_sql)
        item["gold"] = _replace_multiple_spaces(item.get("SQL", gold_sql))
        item["db_id"] = item.get("db_id", gold_db_id)
        item["res"] = int(by_index[index]["res"])
        item["error"] = by_index[index].get("error")
        item.pop("SQL", None)
        details.append(item)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(details, file, indent=2, ensure_ascii=False)


def summarize_by_difficulty(
    results: Sequence[Dict[str, Any]],
    metric_key: str,
    scale_percent: bool = False,
) -> Tuple[List[float], List[int], List[str]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for result in results:
        difficulty = result.get("difficulty") or "unknown"
        grouped[difficulty].append(float(result[metric_key]))
        if difficulty != "total":
            grouped["total"].append(float(result[metric_key]))

    levels = [level for level in DIFFICULTY_LEVELS if level in grouped] + ["total"]
    multiplier = 100 if scale_percent else 1
    scores = [(sum(grouped[level]) / len(grouped[level]) * multiplier) if grouped[level] else 0.0 for level in levels]
    counts = [len(grouped[level]) for level in levels]
    return scores, counts, levels


def print_bird_scores(results: Sequence[Dict[str, Any]]) -> None:
    scores, counts, levels = summarize_by_difficulty(results, "res", scale_percent=True)
    template = "{:20} " + " ".join(["{:<20}"] * len(levels))
    print(template.format("", *levels))
    print(template.format("count", *counts))
    print("====================================== ACCURACY =====================================")
    score_template = "{:20} " + " ".join(["{:<20.2f}"] * len(scores))
    print(score_template.format("accuracy", *scores))


def print_bird_ves(results: Sequence[Dict[str, Any]]) -> None:
    scores, counts, levels = summarize_by_difficulty(results, "ves")
    template = "{:20} " + " ".join(["{:<20}"] * len(levels))
    print(template.format("", *levels))
    print(template.format("count", *counts))
    print("========================================= VES ========================================")
    score_template = "{:20} " + " ".join(["{:<20.2f}"] * len(scores))
    print(score_template.format("ves", *scores))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BIRD execution accuracy and VES evaluation.")
    parser.add_argument("--pred", "--predicted_sql_path", dest="pred", required=True)
    parser.add_argument("--gold", "--ground_truth_path", dest="gold", required=True)
    parser.add_argument("--db_root_path", required=True)
    parser.add_argument("--diff_json_path", default=None)
    parser.add_argument("--meta_time_out", "--exec_timeout", dest="timeout", type=float, default=30.0)
    parser.add_argument("--etype", choices=("exec", "ves", "all"), default="exec")
    parser.add_argument("--ves_iterations", type=int, default=100)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--progress_bar_for_each_datapoint", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSONL file for per-query results.")
    parser.add_argument("--exec_details_output", default=None, help="Optional MAC-SQL-style EX detail JSON file.")
    args = parser.parse_args()

    output_results = []
    if args.etype in ("exec", "all"):
        exec_results = evaluate_bird_execution(
            pred_path=args.pred,
            gold_path=args.gold,
            db_root_path=args.db_root_path,
            diff_json_path=args.diff_json_path,
            timeout=args.timeout,
            show_progress=args.progress_bar_for_each_datapoint,
            num_cpus=args.num_cpus,
        )
        print_bird_scores(exec_results)
        output_results.extend({"metric": "exec", **result} for result in exec_results)
        if args.exec_details_output:
            save_bird_execution_details(exec_results, args.pred, args.gold, args.diff_json_path, args.exec_details_output)

    if args.etype in ("ves", "all"):
        ves_results = evaluate_bird_ves(
            pred_path=args.pred,
            gold_path=args.gold,
            db_root_path=args.db_root_path,
            diff_json_path=args.diff_json_path,
            iterate_num=args.ves_iterations,
            timeout=args.timeout,
            show_progress=args.progress_bar_for_each_datapoint,
            num_cpus=args.num_cpus,
        )
        print_bird_ves(ves_results)
        output_results.extend({"metric": "ves", **result} for result in ves_results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            for result in output_results:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
