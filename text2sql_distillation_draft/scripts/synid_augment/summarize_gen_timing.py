#!/usr/bin/env python3
"""Summarize per-stage wall-clock timings of the offline SynID data generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Stages that make up the data-generation pipeline (setup is reported but not summed).
GEN_ONLY_STAGES = ("gen_loops", "select_merge")
PIPELINE_STAGES = ("prep_format", "gen_loops", "select_merge", "build_train", "tokenize_train")
SETUP_STAGES = ("setup_download_teacher",)


def fmt(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir: Path = args.run_dir

    stages: dict[str, dict[str, Any]] = {}
    for line in (run_dir / "stage_timings.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            stages[row["stage"]] = row

    def seconds(name: str) -> float:
        return float(stages[name]["seconds"]) if name in stages else 0.0

    gen_timing = read_json(run_dir / "synid_aug_v2_lora_218" / "timing.json")
    select_timing = read_json(run_dir / "synid_aug_v2_lora_218" / "overall_synthetic_timing.json")
    gen_summary = read_json(run_dir / "synid_aug_v2_lora_218" / "summary.json")

    gen_only_s = sum(seconds(name) for name in GEN_ONLY_STAGES)
    pipeline_s = sum(seconds(name) for name in PIPELINE_STAGES)
    setup_s = sum(seconds(name) for name in SETUP_STAGES)
    complete = all(name in stages and stages[name]["returncode"] == 0 for name in PIPELINE_STAGES)

    report: dict[str, Any] = {
        "complete": complete,
        "stages": stages,
        "sum_generation_only_s": gen_only_s,
        "sum_pipeline_s": pipeline_s,
        "setup_not_counted_s": setup_s,
        "generation_only_stages": list(GEN_ONLY_STAGES),
        "pipeline_stages": list(PIPELINE_STAGES),
        "gen_loops_detail": gen_timing,
        "select_merge_detail": select_timing,
    }
    if gen_summary:
        report["accepted"] = gen_summary.get("accepted")
        report["rejected_final"] = gen_summary.get("rejected_final")
    (run_dir / "timing_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = ["# Offline data generation timing", ""]
    if not complete:
        lines += ["**INCOMPLETE RUN** - at least one pipeline stage is missing or failed.", ""]
    lines += ["| Stage | Seconds | h:mm:ss | rc |", "|---|---:|---:|---:|"]
    for name in (*SETUP_STAGES, *PIPELINE_STAGES):
        if name in stages:
            row = stages[name]
            tag = " (setup, not counted)" if name in SETUP_STAGES else ""
            lines.append(f"| {name}{tag} | {row['seconds']:.1f} | {fmt(row['seconds'])} | {row['returncode']} |")
    lines += [
        "",
        f"- Sum, generation only ({' + '.join(GEN_ONLY_STAGES)}): **{gen_only_s:.1f} s ({fmt(gen_only_s)})**",
        f"- Sum, whole pipeline ({' + '.join(PIPELINE_STAGES)}): **{pipeline_s:.1f} s ({fmt(pipeline_s)})**",
        "",
    ]

    if gen_timing:
        startup = gen_timing.get("startup", {})
        lines += [
            "## gen_loops breakdown",
            "",
            f"- total inside script: {gen_timing['total_s']:.1f} s "
            f"(driver wall {seconds('gen_loops'):.1f} s; difference = python/CUDA import and exit)",
            f"- load data: {startup.get('load_data_s', 0):.1f} s",
            f"- merge LoRA into base (CPU): {startup.get('merge_lora_s', 0):.1f} s",
            f"- load vLLM engine: {startup.get('vllm_load_s', 0):.1f} s",
            f"- loops total: {gen_timing['loops_total_s']:.1f} s",
            "",
            "| trial t | n examples | prompt s | generate s | extract s | validate s | other s | loop s | new tokens | tok/s |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in gen_timing["loops"]:
            gen_s = item.get("vllm_generate_s", 0.0)
            tokens = item.get("new_tokens", 0)
            tps = tokens / gen_s if gen_s else 0.0
            lines.append(
                f"| {item['loop']} | {item.get('n_examples', 0)} | {item.get('prompt_s', 0):.2f} "
                f"| {gen_s:.1f} | {item.get('extract_sql_s', 0):.2f} | {item.get('validate_s', 0):.1f} "
                f"| {item.get('other_s', 0):.1f} | {item['loop_total_s']:.1f} | {tokens} | {tps:.0f} |"
            )
        lines += [
            "",
            "Validate sub-steps, summed over examples (s):",
            "",
            "| trial t | exec gold | exec candidate | ExecutionMatch | SequenceMatcher | validate total |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for item in gen_timing["loops"]:
            lines.append(
                f"| {item['loop']} | {item.get('sum_exec_gold_s', 0):.1f} | {item.get('sum_exec_cand_s', 0):.1f} "
                f"| {item.get('sum_exec_match_s', 0):.1f} | {item.get('sum_similarity_s', 0):.2f} "
                f"| {item.get('sum_validate_total_s', 0):.1f} |"
            )
        env = gen_timing.get("environment", {})
        lines += ["", f"Environment: `{json.dumps(env)}`", ""]

    if select_timing:
        lines += [
            "## select_merge breakdown",
            "",
            "- " + ", ".join(f"{k}={v:.2f}s" for k, v in select_timing.items() if k.endswith("_s")),
            "",
        ]

    (run_dir / "timing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
