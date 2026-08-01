import argparse
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
        "table": "benchmarks/spider_data/tables.json",
    },
    "spider_realistic_test": {
        "gold": None,
        "db": "benchmarks/spider_data/database",
        "table": "benchmarks/spider_data/tables.json",
    },
    "spider_dk_test": {
        "gold": None,
        "db": "benchmarks/spider_dk/database",
        "table": "benchmarks/spider_dk/tables.json",
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
        choices=("all", "exec", "match"),
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
