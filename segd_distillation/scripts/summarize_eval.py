#!/usr/bin/env python3
"""Aggregate MMEB-eval *_score.json files into a table (+ optional text dump).

Works for any metatask (cls / VQA / retrieval / …) as long as eval wrote
``*_score.json`` under the output directory.

Usage:
  python scripts/summarize_eval.py eval_outputs/EXP_NAME
  python scripts/summarize_eval.py eval_outputs/EXP_NAME -o results/EXP_summary.txt
  python scripts/summarize_eval.py eval_outputs/EXP_NAME \\
      --meta exp=... --meta ckpt=... --meta subset=all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _flatten_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k, v in data.items():
        if isinstance(v, (int, float)):
            metrics[str(k)] = float(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    metrics[f"{k}.{kk}"] = float(vv)
    return metrics


def collect_rows(eval_dir: Path) -> List[Tuple[str, Dict[str, float], Path]]:
    files = sorted({*eval_dir.glob("*_score.json"), *eval_dir.rglob("*_score.json")})
    if not files:
        files = sorted(p for p in eval_dir.rglob("*.json") if "score" in p.name.lower())

    rows: List[Tuple[str, Dict[str, float], Path]] = []
    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"skip {path}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        metrics = _flatten_metrics(data)
        if not metrics:
            continue
        subset = (
            data.get("subset")
            or data.get("dataset")
            or path.stem.replace("_score", "")
        )
        rows.append((str(subset), metrics, path))
    return rows


def format_table(
    rows: List[Tuple[str, Dict[str, float], Path]],
    meta: Optional[Dict[str, str]] = None,
) -> str:
    lines: List[str] = []
    if meta:
        for k, v in meta.items():
            lines.append(f"{k}={v}")
        lines.append("")

    if not rows:
        lines.append("(no score json found)")
        return "\n".join(lines) + "\n"

    keys: List[str] = []
    for _, metrics, _ in rows:
        for k in metrics:
            if k not in keys:
                keys.append(k)

    hdr = f"{'subset':<28}" + "".join(f"{k:>12}" for k in keys)
    lines.append(hdr)
    lines.append("-" * len(hdr))

    acc = {k: [] for k in keys}
    for subset, metrics, _ in rows:
        line = f"{subset:<28}"
        for k in keys:
            if k in metrics:
                line += f"{metrics[k]:12.4f}"
                acc[k].append(metrics[k])
            else:
                line += f"{'—':>12}"
        lines.append(line)

    if keys:
        line = f"{'MEAN':<28}"
        for k in keys:
            vals = acc[k]
            mean = sum(vals) / len(vals) if vals else float("nan")
            line += f"{mean:12.4f}"
        lines.append(line)

    lines.append("")
    lines.append(f"n_files={len(rows)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MMEB-eval score JSON files")
    parser.add_argument("eval_dir", type=Path, help="Directory with *_score.json")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Write summary text here (also prints to stdout)",
    )
    parser.add_argument(
        "--meta", action="append", default=[],
        help="Extra header line KEY=VALUE (repeatable)",
    )
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        raise SystemExit(f"not a directory: {args.eval_dir}")

    meta: Dict[str, str] = {}
    for item in args.meta:
        if "=" not in item:
            raise SystemExit(f"--meta expects KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        meta[k] = v

    rows = collect_rows(args.eval_dir)
    text = format_table(rows, meta=meta or None)
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
