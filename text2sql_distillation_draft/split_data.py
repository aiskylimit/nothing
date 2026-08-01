"""
split_data.py – Split train.jsonl → train.jsonl (80%) + dev.jsonl (20%)
========================================================================

Finds every ``train.jsonl`` under ``datasets/train/*/format_data/`` and splits
it in-place. The original file is overwritten with the train portion;
``dev.jsonl`` is written alongside it.

Usage
-----
# Split all training datasets (default)
python split_data.py

# Split a specific training dataset only
python split_data.py --benchmark Cypherbench

# Custom ratio (e.g. 90/10)
python split_data.py --train-ratio 0.9

# Fix the random seed for reproducibility
python split_data.py --seed 42
"""

import random
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split train.jsonl into train (80%) and dev (20%) sets"
    )
    parser.add_argument(
        "--base-dir",
        default="",
        help="Project root directory",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help=(
            "Training dataset to split (e.g. spider_data). "
            "If omitted, all datasets under datasets/train/ are processed."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data to keep as training set (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling (preserve original order)",
    )
    return parser.parse_args()


def split_file(train_jsonl: Path, train_ratio: float, seed: int, shuffle: bool):
    """Read *source* jsonl, split, and write train.jsonl + dev.jsonl."""
    lines = train_jsonl.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]  # drop blank lines

    if not lines:
        print(f"  [SKIP] {train_jsonl} is empty.")
        return

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(lines)

    split_idx = max(1, int(len(lines) * train_ratio))
    train_lines = lines[:split_idx]
    dev_lines = lines[split_idx:]

    out_dir = train_jsonl.parent
    out_train = out_dir / "train.jsonl"
    out_dev   = out_dir / "dev.jsonl"

    # Write train split (may overwrite the source if it was already train.jsonl)
    out_train.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    # Write dev split
    out_dev.write_text("\n".join(dev_lines) + "\n", encoding="utf-8")

    src_note = f" (from {train_jsonl.name})" if train_jsonl.name != "train.jsonl" else ""
    print(
        f"  [OK] {out_dir.name}{src_note}: "
        f"{len(train_lines)} train / {len(dev_lines)} dev "
        f"(total {len(lines)})"
    )


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    datasets_dir = base_dir / "datasets" / "train"

    if args.benchmark:
        benchmark_dirs = [datasets_dir / args.benchmark / "format_data"]
    else:
        benchmark_dirs = sorted(
            p / "format_data" for p in datasets_dir.iterdir() if p.is_dir()
        )

    print(
        f"Split ratio: {args.train_ratio:.0%} train / {1 - args.train_ratio:.0%} dev  |  "
        f"seed={args.seed}  |  shuffle={'no' if args.no_shuffle else 'yes'}\n"
    )

    found_any = False
    for bdir in benchmark_dirs:
        src_filename = "train.jsonl"
        src_jsonl = bdir / src_filename
        if not src_jsonl.exists():
            print(
                f"  [SKIP] {bdir.name}: {src_filename} not found "
                f"(run format_data.py first)"
            )
            continue
        found_any = True
        print(f"Processing {bdir.name} (source: {src_filename}) ...")
        split_file(
            train_jsonl=src_jsonl,
            train_ratio=args.train_ratio,
            seed=args.seed,
            shuffle=not args.no_shuffle,
        )

    if not found_any:
        print(
            "\nNo train.jsonl files found. "
            "Run format_data.py first to generate them."
        )
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
