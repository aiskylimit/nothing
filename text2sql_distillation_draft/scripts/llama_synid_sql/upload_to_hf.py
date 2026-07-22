#!/usr/bin/env python3
"""Upload Llama SynID-SQL checkpoints, inference outputs, and eval outputs to HF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_UPLOADS = [
    ("results/llama_synid_sql", "llama/synid_sql/checkpoints"),
    ("results/infer/llama_synid_sql", "llama/synid_sql/infer"),
    ("results/eval/llama_synid_sql", "llama/synid_sql/eval"),
]


def upload_folder(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    folder_path: Path,
    path_in_repo: str,
    dry_run: bool,
) -> None:
    if not folder_path.is_dir():
        print(f"[upload-skip] missing folder: {folder_path}")
        return

    print(f"[upload] {folder_path} -> {repo_id}/{path_in_repo}")
    if dry_run:
        return

    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=f"Upload Llama SynID-SQL artifacts: {path_in_repo}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", "distillation-sql/llama_spider"))
    parser.add_argument("--repo-type", default=os.environ.get("HF_REPO_TYPE", "model"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    for local_path, path_in_repo in DEFAULT_UPLOADS:
        upload_folder(
            api=api,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=Path(local_path),
            path_in_repo=path_in_repo,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
