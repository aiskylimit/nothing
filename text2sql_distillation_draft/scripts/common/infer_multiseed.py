#!/usr/bin/env python3
"""Run infer.py for the five reporting seeds."""

from __future__ import annotations

import gc
import json
import os
import random
import sys
from pathlib import Path

import torch
from transformers import set_seed as transformers_set_seed

import infer


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
        raise ValueError("No valid seeds found.")
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


def value_after_flag(args: list[str], flag: str) -> str:
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return ""


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


def write_meta(seed: int, args: list[str], output_path: Path | None) -> None:
    if output_path is None:
        return
    meta = {
        "seed": seed,
        "benchmark": value_after_flag(args, "--benchmark"),
        "split": value_after_flag(args, "--split"),
        "model": value_after_flag(args, "--model"),
        "ckpt_path": value_after_flag(args, "--ckpt_path"),
        "output_path": str(output_path),
    }
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    seeds = parse_seeds(os.environ.get("INFER_SEEDS", "10,42,50,100,1234"))
    base_args = sys.argv[1:]
    for seed in seeds:
        set_all_seeds(seed)
        seeded_args, output_path = replace_output_path(base_args, seed)
        print(f"[infer] seed={seed} output={output_path}", flush=True)
        old_argv = sys.argv
        try:
            sys.argv = ["infer.py", *seeded_args]
            infer.main()
            write_meta(seed, seeded_args, output_path)
        finally:
            sys.argv = old_argv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
