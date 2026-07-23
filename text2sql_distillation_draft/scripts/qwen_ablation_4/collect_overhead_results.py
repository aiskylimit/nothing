#!/usr/bin/env python3
"""Collect ablation-4 overhead metrics into JSON and LaTeX rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


METHODS = [
    ("distillm", "DistiLLM", "qwen_ablation_4_distillm"),
    ("distillm_dcr", r"\quad \textit{w/} $\mathcal{L}_{\mathrm{DCR}}$", "qwen_ablation_4_distillm_dcr"),
    ("csd", "CSD", "qwen_ablation_4_csd"),
    ("csd_dcr", r"\quad \textit{w/} $\mathcal{L}_{\mathrm{DCR}}$", "qwen_ablation_4_csd_dcr"),
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict], last_epoch_only: bool) -> dict | None:
    if not rows:
        return None
    if last_epoch_only:
        rows = [rows[-1]]
    return {
        "epochs": [int(row["epoch"]) for row in rows],
        "time_step_s": mean(step_time(row) for row in rows),
        "avg_alloc_gb": mean(float(row["avg_alloc_gb"]) for row in rows),
        "peak_alloc_gb": mean(float(row["peak_alloc_gb"]) for row in rows),
    }


def step_time(row: dict) -> float:
    if "time_step_s" in row:
        return float(row["time_step_s"])
    num_steps = int(row.get("num_steps", 0))
    if num_steps <= 0:
        raise ValueError(
            "Cannot derive time/step from old overhead row without num_steps: "
            f"{row}"
        )
    return float(row["time_epoch_s"]) / num_steps


def fmt(value: float | None) -> str:
    if value is None:
        return r"\NA"
    return f"{value:.2f}"


def make_table(results: dict[str, dict | None]) -> str:
    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Time/step} & \textbf{Avg. alloc.} & \textbf{Peak alloc.} \\")
    lines.append(r"& (s) & (GB) & (GB) \\")
    lines.append(r"\midrule")
    for idx, (key, label, _) in enumerate(METHODS):
        summary = results.get(key)
        time_value = fmt(None if summary is None else summary["time_step_s"])
        avg_value = fmt(None if summary is None else summary["avg_alloc_gb"])
        peak_value = fmt(None if summary is None else summary["peak_alloc_gb"])
        lines.append(f"{label} & {time_value} & {avg_value} & {peak_value} \\\\")
        if key == "distillm_dcr":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root", type=Path, default=Path("results/qwen3"))
    parser.add_argument("--output-root", type=Path, default=Path("results/overhead/qwen_ablation_4"))
    parser.add_argument("--last-epoch-only", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict | None] = {}
    sources: dict[str, str] = {}

    for key, _, save_tag in METHODS:
        metric_path = args.save_root / save_tag / "overhead_metrics.jsonl"
        if metric_path.exists():
            results[key] = summarize(read_jsonl(metric_path), args.last_epoch_only)
            sources[key] = str(metric_path)
        else:
            results[key] = None
            sources[key] = str(metric_path)

    payload = {
        "results": results,
        "sources": sources,
        "last_epoch_only": args.last_epoch_only,
    }
    (args.output_root / "computational_overhead.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "computational_overhead_table.tex").write_text(
        make_table(results),
        encoding="utf-8",
    )
    print(f"[collect] wrote {args.output_root / 'computational_overhead.json'}")
    print(f"[collect] wrote {args.output_root / 'computational_overhead_table.tex'}")


if __name__ == "__main__":
    main()
