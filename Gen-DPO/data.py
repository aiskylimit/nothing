#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

def convert_value(val):
    """Chuyển đổi mọi kiểu numpy/pandas sang kiểu Python thuần."""
    import numpy as np
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return [convert_value(v) for v in val.tolist()]
    if isinstance(val, list):
        return [convert_value(v) for v in val]
    if isinstance(val, dict):
        return {k: convert_value(v) for k, v in val.items()}
    return val

def parquet_to_jsonl(input_path: str, output_path: str = None):
    try:
        import pandas as pd
    except ImportError:
        print("Thiếu thư viện pandas. Cài bằng: pip install pandas pyarrow")
        sys.exit(1)

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Lỗi: Không tìm thấy file '{input_path}'")
        sys.exit(1)

    if output_path is None:
        output_path = input_path.with_suffix(".jsonl")
    output_path = Path(output_path)

    print(f"Đang đọc: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Đã đọc xong: {df.shape[0]:,} hàng × {df.shape[1]} cột")
    print(f"Cột: {df.columns.tolist()}")

    print(f"Đang ghi: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {col: convert_value(row[col]) for col in df.columns}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Hoàn thành! File output: {output_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển Parquet → JSONL")
    parser.add_argument(
        "--input", "-i",
        default="/workspace/ComfyUI/models/instantid/Gen-DPO/datasets/deita/data/test_sft-00000-of-00001.parquet",
    )
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    parquet_to_jsonl(args.input, args.output)