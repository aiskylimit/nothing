#!/usr/bin/env python3
"""Collect SynID no-keywords-weight grid eval logs into per-seed JSON summaries."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KD_RATIOS = [0.7, 0.6]
LAYER_CONFIGS = [
    ("last1", 1, "27", "35"),
    ("last3", 3, "25,26,27", "33,34,35"),
    ("near_last3", 3, "24,25,26", "32,33,34"),
    ("near_last3_prev", 3, "23,24,25", "31,32,33"),
]
DEFAULT_BENCHMARKS = ["spider_data", "spider_syn", "spider_realistic", "spider_dk"]


def grid_config(grid_id: str) -> dict[str, Any]:
    match = re.fullmatch(r"G(\d+)", grid_id)
    if not match:
        raise ValueError(f"invalid grid_id: {grid_id}")
    grid_num = int(match.group(1))
    index = grid_num - 1
    if index < 0 or index >= len(LAYER_CONFIGS) * len(KD_RATIOS):
        raise ValueError(f"grid_id out of range: {grid_id}")
    kd_index = index % len(KD_RATIOS)
    layer_name, k, student_layers, teacher_layers = LAYER_CONFIGS[index // len(KD_RATIOS)]
    return {
        "grid_id": grid_id,
        "k": k,
        "layer_config": layer_name,
        "student_layers": student_layers,
        "teacher_layers": teacher_layers,
        "alpha": 0.1,
        "kd_ratio": KD_RATIOS[kd_index],
        "beta": 0.1,
        "contrastive_tau": 0.05,
        "pooling": "sc",
        "use_syntax_weights": False,
        "syntax_lambda": 1.0,
    }


def parse_grid_id(run_name: str) -> str:
    match = re.search(r"train_g(\d+)", run_name)
    if not match:
        return ""
    return f"G{int(match.group(1)):02d}"


def parse_scores(log_path: Path) -> dict[str, float | None]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    levels: list[str] = []
    in_main_exec = False
    in_main_exact = False
    scores: dict[str, float | None] = {
        "Exact Match": None,
        "Execution Accuracy": None,
    }

    for line in lines:
        tokens = line.split()
        if tokens[:5] == ["easy", "medium", "hard", "extra", "all"]:
            levels = tokens
            continue
        if "TURN EXECUTION ACCURACY" in line or "TURN EXACT MATCHING ACCURACY" in line:
            in_main_exec = False
            in_main_exact = False
            continue
        if "EXECUTION ACCURACY" in line:
            in_main_exec = True
            in_main_exact = False
            continue
        if "EXACT MATCHING ACCURACY" in line:
            in_main_exact = True
            in_main_exec = False
            continue

        if in_main_exec and tokens and tokens[0] == "execution":
            values = [float(item) for item in tokens[1:]]
            scores["Execution Accuracy"] = score_for_all(levels, values)
            in_main_exec = False
        elif in_main_exact and len(tokens) >= 3 and tokens[0] == "exact" and tokens[1] == "match":
            values = [float(item) for item in tokens[2:]]
            scores["Exact Match"] = score_for_all(levels, values)
            in_main_exact = False

    return scores


def score_for_all(levels: list[str], values: list[float]) -> float | None:
    if not values:
        return None
    if "all" in levels and len(values) > levels.index("all"):
        return values[levels.index("all")]
    return values[-1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def meta_path_for(infer_seed_dir: Path, run_name: str) -> Path:
    return infer_seed_dir / f"{run_name}__full_sql_result.eval_meta.json"


def artifact_paths(
    *,
    infer_output_root: Path,
    benchmark: str,
    seed_name: str,
    run_name: str,
    pred_path: Path,
    gold_path: Path,
    log_path: Path,
) -> dict[str, str]:
    infer_seed_dir = infer_output_root / benchmark / seed_name
    meta = read_json(meta_path_for(infer_seed_dir, run_name))
    return {
        "run_name": run_name,
        "pred_path": str(pred_path),
        "gold_path": str(gold_path),
        "eval_log_path": str(log_path),
        "ckpt_path": str(meta.get("ckpt_path", "")),
    }


def build_result(
    *,
    infer_output_root: Path,
    eval_output_root: Path,
    benchmark: str,
    seed_dir: Path,
    log_path: Path,
) -> dict[str, Any] | None:
    seed_name = seed_dir.name
    seed_match = re.fullmatch(r"seed(\d+)", seed_name)
    if not seed_match:
        return None
    seed = int(seed_match.group(1))
    run_name = log_path.name.split(".etype-", 1)[0]
    grid_id = parse_grid_id(run_name)
    if not grid_id:
        return None

    infer_seed_dir = infer_output_root / benchmark / seed_name
    formatted_dir = infer_seed_dir / "formatted_data"
    pred_path = formatted_dir / f"{run_name}.pred.sql"
    gold_path = formatted_dir / f"{run_name}.gold.sql"

    result = {
        "grid": grid_config(grid_id),
        "benchmark": benchmark,
        "seed": seed,
        "scores": parse_scores(log_path),
        "artifacts": artifact_paths(
            infer_output_root=infer_output_root,
            benchmark=benchmark,
            seed_name=seed_name,
            run_name=run_name,
            pred_path=pred_path,
            gold_path=gold_path,
            log_path=log_path,
        ),
    }
    return result


def score_key(result: dict[str, Any]) -> tuple[float, float]:
    scores = result.get("scores", {})
    exec_score = scores.get("Execution Accuracy")
    exact_score = scores.get("Exact Match")
    return (
        float(exec_score) if exec_score is not None else -1.0,
        float(exact_score) if exact_score is not None else -1.0,
    )


def best_by_benchmark(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["benchmark"]].append(result)

    best: dict[str, Any] = {}
    for benchmark, items in sorted(grouped.items()):
        winner = max(items, key=score_key)
        best[benchmark] = {
            "grid": winner["grid"],
            "scores": winner["scores"],
            "artifacts": winner["artifacts"],
        }
    return best


def best_overall(results: list[dict[str, Any]], required_benchmarks: list[str]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        grouped[result["grid"]["grid_id"]][result["benchmark"]] = result

    candidates = []
    for grid_id, by_benchmark in grouped.items():
        if any(benchmark not in by_benchmark for benchmark in required_benchmarks):
            continue
        selected = [by_benchmark[benchmark] for benchmark in required_benchmarks]
        mean_exec = mean_score(selected, "Execution Accuracy")
        mean_exact = mean_score(selected, "Exact Match")
        candidates.append((mean_exec, mean_exact, grid_id, selected))

    if not candidates:
        return {
            "best": None,
            "reason": "No grid has all required benchmarks.",
        }

    mean_exec, mean_exact, grid_id, selected = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "best": {
            "grid": selected[0]["grid"],
            "mean_scores": {
                "Exact Match": mean_exact,
                "Execution Accuracy": mean_exec,
            },
            "per_benchmark": {
                item["benchmark"]: item["scores"]
                for item in selected
            },
            "artifacts": {
                item["benchmark"]: item["artifacts"]
                for item in selected
            },
        }
    }


def mean_score(results: list[dict[str, Any]], key: str) -> float:
    values = [item["scores"].get(key) for item in results]
    present = [float(value) for value in values if value is not None]
    if not present:
        return -1.0
    return round(sum(present) / len(present), 6)


def write_seed_outputs(
    *,
    eval_output_root: Path,
    seed: int,
    results: list[dict[str, Any]],
    required_benchmarks: list[str],
) -> None:
    seed_out_dir = eval_output_root / f"seed{seed}"
    seed_out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    full_payload = {
        "seed": seed,
        "generated_at": generated_at,
        "selection": {
            "primary_metric": "Execution Accuracy",
            "tie_breaker": "Exact Match",
        },
        "results": sorted(
            results,
            key=lambda item: (item["grid"]["grid_id"], item["benchmark"]),
        ),
    }
    (seed_out_dir / "eval_grid_results.json").write_text(
        json.dumps(full_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    by_benchmark_payload = {
        "seed": seed,
        "generated_at": generated_at,
        "primary_metric": "Execution Accuracy",
        "tie_breaker": "Exact Match",
        "best_by_benchmark": best_by_benchmark(results),
    }
    (seed_out_dir / "best_grid_by_benchmark.json").write_text(
        json.dumps(by_benchmark_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    overall_payload = {
        "seed": seed,
        "generated_at": generated_at,
        "aggregation": {
            "benchmarks": required_benchmarks,
            "primary_metric": "mean Execution Accuracy",
            "tie_breaker": "mean Exact Match",
        },
        **best_overall(results, required_benchmarks),
    }
    (seed_out_dir / "best_grid_overall.json").write_text(
        json.dumps(overall_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def collect(args: argparse.Namespace) -> int:
    eval_output_root = args.eval_output_root
    infer_output_root = args.infer_output_root
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for benchmark_dir in sorted(eval_output_root.iterdir() if eval_output_root.exists() else []):
        if not benchmark_dir.is_dir() or benchmark_dir.name.startswith("seed"):
            continue
        benchmark = benchmark_dir.name
        for seed_dir in sorted(benchmark_dir.glob("seed*")):
            if not seed_dir.is_dir():
                continue
            for log_path in sorted(seed_dir.glob("*.etype-*.log")):
                result = build_result(
                    infer_output_root=infer_output_root,
                    eval_output_root=eval_output_root,
                    benchmark=benchmark,
                    seed_dir=seed_dir,
                    log_path=log_path,
                )
                if result is not None:
                    by_seed[result["seed"]].append(result)

    for seed, results in sorted(by_seed.items()):
        write_seed_outputs(
            eval_output_root=eval_output_root,
            seed=seed,
            results=results,
            required_benchmarks=args.required_benchmarks,
        )
        print(f"[collect] seed{seed}: {len(results)} result(s)")

    if not by_seed:
        print("[collect] no eval logs found")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--infer-output-root",
        type=Path,
        default=Path("results/infer/synid_ce_no_keywords_weight_lora/qwen"),
    )
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=Path("results/eval/synid_ce_no_keywords_weight_lora/qwen"),
    )
    parser.add_argument(
        "--required-benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
    )
    args = parser.parse_args()
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
