import argparse
import json
import os
import sys


BENCHMARKS = {
    "spider_dev": {
        "gold": "benchmarks/spider_data/dev_gold.sql",
        "db": "benchmarks/spider_data/database",
        "table": "benchmarks/spider_data/tables.json",
    },
    "spider_test": {
        "gold": "benchmarks/spider_data/test_gold.sql",
        "db": "benchmarks/spider_data/test_database",
        "table": "benchmarks/spider_data/test_tables.json",
    },
    "spider_syn_test": {
        "gold": None,
        "db": "benchmarks/spider_data/database",
        "table": "benchmarks_2/spider_data/tables.json",
    },
    "spider_realistic_test": {
        "gold": None,
        "db": "benchmarks/spider_data/database",
        "table": "benchmarks_2/spider_data/tables.json",
    },
    "spider_dk_test": {
        "gold": None,
        "db": "benchmarks/spider_dk/database",
        "table": "benchmarks_2/spider_dk/tables.json",
    },
    "bird_dev": {
        "gold": "benchmarks_2/bird/dev/dev.sql",
        "db": "benchmarks/bird/dev/dev_databases",
        "table": "benchmarks_2/bird/dev/dev_tables.json",
        "diff": "benchmarks_2/bird/dev/dev.json",
    },
    "sparc_dev": {
        "gold": "benchmarks/sparc/dev_gold.txt",
        "db": "benchmarks/sparc/database",
        "table": "benchmarks/sparc/tables.json",
    },
    "cosql_dev": {
        "gold": "benchmarks/cosql_dataset/sql_state_tracking/dev_gold.txt",
        "db": "benchmarks/cosql_dataset/database",
        "table": "benchmarks/cosql_dataset/tables.json",
    },
}


def read_sessions(path):
    sessions = []
    current = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if len(line.strip()) == 0:
                sessions.append(current)
                current = []
            else:
                current.append(line.rstrip("\n"))
    if current:
        sessions.append(current)
    return sessions


def validate_prediction_shape(gold_path, pred_path):
    gold_sessions = read_sessions(gold_path)
    pred_sessions = read_sessions(pred_path)

    if len(gold_sessions) != len(pred_sessions):
        raise ValueError(
            "Prediction session count does not match gold: "
            f"{len(pred_sessions)} != {len(gold_sessions)}"
        )

    for idx, (gold_turns, pred_turns) in enumerate(zip(gold_sessions, pred_sessions), start=1):
        if len(gold_turns) != len(pred_turns):
            raise ValueError(
                f"Prediction turn count does not match gold in session {idx}: "
                f"{len(pred_turns)} != {len(gold_turns)}"
            )


def resolve_repo_path(path):
    return os.path.abspath(path)


def main():
    parser = argparse.ArgumentParser(description="Run text-to-SQL benchmark evaluation.")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(BENCHMARKS),
        help="Benchmark split to evaluate.",
    )
    parser.add_argument("--pred", required=True, help="Prediction file path.")
    parser.add_argument(
        "--gold",
        default=None,
        help="Gold SQL file path. Required for benchmark configs without a built-in gold file.",
    )
    parser.add_argument(
        "--etype",
        default="exec",
        choices=("all", "exec", "match", "ves"),
        help="Evaluation type.",
    )
    parser.add_argument(
        "--plug_value",
        default=False,
        action="store_true",
        help="Plug gold values into predicted SQL before execution evaluation.",
    )
    parser.add_argument(
        "--keep_distinct",
        default=False,
        action="store_true",
        help="Keep DISTINCT during evaluation.",
    )
    parser.add_argument(
        "--progress_bar_for_each_datapoint",
        default=False,
        action="store_true",
        help="Show per-datapoint database execution progress bars.",
    )
    parser.add_argument(
        "--exec_timeout",
        type=int,
        default=None,
        help="Maximum seconds allowed for each SQL execution before marking it wrong.",
    )
    parser.add_argument(
        "--ves_iterations",
        type=int,
        default=100,
        help="Number of repeated timing runs for BIRD VES evaluation.",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of worker processes for BIRD EX/VES evaluation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSONL file for BIRD per-query metric results.",
    )
    parser.add_argument(
        "--exec_details_output",
        default=None,
        help="Optional MAC-SQL-style JSON detail file for BIRD EX results.",
    )
    parser.add_argument(
        "--skip_shape_check",
        default=False,
        action="store_true",
        help="Skip validation that pred has the same session/turn shape as gold.",
    )
    parser.add_argument(
        "--check_only",
        default=False,
        action="store_true",
        help="Only validate benchmark paths and prediction shape, then exit.",
    )
    args = parser.parse_args()

    config = BENCHMARKS[args.benchmark]
    gold_config = args.gold or config["gold"]
    if gold_config is None:
        raise ValueError(
            f"{args.benchmark} does not have a built-in gold file. "
            "Pass --gold, usually the .gold.sql file created by scripts/format_spider_infer_results.py."
        )
    gold_path = resolve_repo_path(gold_config)
    pred_path = resolve_repo_path(args.pred)
    db_dir = resolve_repo_path(config["db"])
    table_path = resolve_repo_path(config["table"])

    for path_name, path in (
        ("gold", gold_path),
        ("pred", pred_path),
        ("db", db_dir),
        ("table", table_path),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name} path does not exist: {path}")

    if not args.skip_shape_check:
        validate_prediction_shape(gold_path, pred_path)

    if args.check_only:
        print("Prediction shape is valid.")
        return 0

    if args.benchmark.startswith("bird"):
        if args.etype == "match":
            raise ValueError("BIRD does not use Spider exact-match parsing; use --etype exec, ves, or all.")
        from bird_evaluation import (
            evaluate_bird_execution,
            evaluate_bird_ves,
            print_bird_scores,
            print_bird_ves,
            save_bird_execution_details,
        )

        diff_path = resolve_repo_path(config["diff"]) if config.get("diff") else None
        output_results = []
        if args.etype in ("exec", "all"):
            results = evaluate_bird_execution(
                pred_path=pred_path,
                gold_path=gold_path,
                db_root_path=db_dir,
                diff_json_path=diff_path,
                timeout=args.exec_timeout,
                show_progress=args.progress_bar_for_each_datapoint,
                num_cpus=args.num_cpus,
            )
            print_bird_scores(results)
            output_results.extend({"metric": "exec", **result} for result in results)
            details_output = args.exec_details_output or os.path.join(os.path.dirname(pred_path), "eval_result_dev.json")
            save_bird_execution_details(results, pred_path, gold_path, diff_path, details_output)
            print(f"save BIRD EX details to {details_output}")
        if args.etype in ("ves", "all"):
            results = evaluate_bird_ves(
                pred_path=pred_path,
                gold_path=gold_path,
                db_root_path=db_dir,
                diff_json_path=diff_path,
                iterate_num=args.ves_iterations,
                timeout=args.exec_timeout,
                show_progress=args.progress_bar_for_each_datapoint,
                num_cpus=args.num_cpus,
            )
            print_bird_ves(results)
            output_results.extend({"metric": "ves", **result} for result in results)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as file:
                for result in output_results:
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"save BIRD metric results to {args.output}")
        return 0

    if args.etype == "ves":
        raise ValueError("VES is only implemented for BIRD benchmarks.")

    try:
        from evaluation import evaluate, build_foreign_key_map_from_json
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing evaluator dependency: {exc.name}. "
            "Install/sync project dependencies from pyproject.toml before running evaluation."
        ) from exc

    kmaps = None
    if args.etype in ("all", "match"):
        kmaps = build_foreign_key_map_from_json(table_path)

    evaluate(
        gold=gold_path,
        predict=pred_path,
        db_dir=db_dir,
        etype=args.etype,
        kmaps=kmaps,
        plug_value=args.plug_value,
        keep_distinct=args.keep_distinct,
        progress_bar_for_each_datapoint=args.progress_bar_for_each_datapoint,
        exec_timeout=args.exec_timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
