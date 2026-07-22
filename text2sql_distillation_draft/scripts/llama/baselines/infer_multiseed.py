#!/usr/bin/env python3
"""Run infer.py repeatedly with deterministic seeds for Llama baselines."""

from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
from transformers import set_seed as transformers_set_seed

import infer


DEFAULT_SEEDS = "10,42,50,100,1234"


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for item in raw.replace(";", ",").split(","):
        text = item.strip()
        if not text:
            continue
        if text.lower().startswith("seed"):
            text = text[4:]
        seeds.append(int(text))
    if not seeds:
        raise ValueError("No valid seeds found in INFER_SEEDS.")
    return seeds


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)


def replace_output_path(args: list[str], seed: int) -> tuple[list[str], Path | None]:
    next_args = list(args)
    for idx, arg in enumerate(next_args):
        if arg == "--output_path" and idx + 1 < len(next_args):
            base = Path(next_args[idx + 1])
            seeded = base.parent / f"seed{seed}" / base.name
            seeded.parent.mkdir(parents=True, exist_ok=True)
            next_args[idx + 1] = str(seeded)
            return next_args, seeded
        if arg.startswith("--output_path="):
            base = Path(arg.split("=", 1)[1])
            seeded = base.parent / f"seed{seed}" / base.name
            seeded.parent.mkdir(parents=True, exist_ok=True)
            next_args[idx] = f"--output_path={seeded}"
            return next_args, seeded
    return next_args, None


def format_after_infer(output_path: Path | None) -> None:
    if output_path is None:
        return
    if os.environ.get("FORMAT_AFTER_INFER", "false").lower() not in {"1", "true", "yes", "y"}:
        return
    print(f"[format-start] output={output_path}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/format_spider_infer_results.py",
            "--input-dir",
            str(output_path.parent),
            "--input-glob",
            output_path.name,
        ],
        check=True,
    )
    print(f"[format-done] output={output_path}", flush=True)


def value_after_flag(args: list[str], flag: str) -> str:
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return ""


def write_eval_meta(seed: int, args: list[str], output_path: Path | None) -> None:
    if output_path is None:
        return
    meta_path = output_path.with_suffix(".eval_meta.json")
    meta = {
        "seed": seed,
        "benchmark": value_after_flag(args, "--benchmark"),
        "split": value_after_flag(args, "--split"),
        "db": value_after_flag(args, "--db"),
        "output_path": str(output_path),
        "train_script": os.environ.get("INFER_TRAIN_SCRIPT", ""),
        "ckpt_path": os.environ.get("INFER_CKPT_PATH", ""),
        "ckpt_step": os.environ.get("INFER_CKPT_STEP", ""),
        "checkpoint_metric": os.environ.get("INFER_CHECKPOINT_METRIC", ""),
        "checkpoint_selection_log": os.environ.get("INFER_CHECKPOINT_SELECTION_LOG", ""),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def should_skip(output_path: Path | None) -> bool:
    if output_path is None:
        return False
    if os.environ.get("SKIP_EXISTING", "false").lower() not in {"1", "true", "yes", "y"}:
        return False
    return output_path.is_file() and output_path.stat().st_size > 0


def main() -> None:
    seeds = parse_seeds(os.environ.get("INFER_SEEDS", DEFAULT_SEEDS))
    base_args = sys.argv[1:]

    for seed in seeds:
        print(f"[infer-seed] seed={seed}", flush=True)
        set_all_seeds(seed)
        seeded_args, output_path = replace_output_path(base_args, seed)
        if should_skip(output_path):
            write_eval_meta(seed, seeded_args, output_path)
            print(f"[infer-seed-skip] seed={seed} output={output_path}", flush=True)
            continue
        print(f"[infer-seed-start] seed={seed} output={output_path}", flush=True)
        old_argv = sys.argv
        try:
            sys.argv = ["infer.py", *seeded_args]
            infer.main()
            write_eval_meta(seed, seeded_args, output_path)
            format_after_infer(output_path)
            print(f"[infer-seed-done] seed={seed} output={output_path}", flush=True)
        finally:
            sys.argv = old_argv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
