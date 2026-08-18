from __future__ import annotations

import json
import time
from pathlib import Path

from transformers import TrainerCallback


class LocalJSONLCallback(TrainerCallback):
    """Persist Trainer metrics locally without an external reporter."""

    def __init__(self, filename: str = "training_log.jsonl") -> None:
        self.filename = filename
        self.started_at = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.time()
        if state.is_world_process_zero and state.global_step == 0:
            self._path(args).unlink(missing_ok=True)
        if state.is_world_process_zero:
            self._write_time_summary(args, state, status="running")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return

        path = self._path(args)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "step": state.global_step,
            "elapsed_seconds": round(time.time() - self.started_at, 3),
            **logs,
        }
        with path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=True)
            handle.write("\n")
        self._write_time_summary(args, state, status="running")

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write_time_summary(args, state, status="completed")

    def _path(self, args) -> Path:
        return Path(args.output_dir) / self.filename

    def _write_time_summary(self, args, state, status: str) -> None:
        elapsed_seconds = time.time() - self.started_at
        summary = {
            "status": status,
            "completed_steps": state.global_step,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "elapsed_hours": round(elapsed_seconds / 3600, 6),
        }
        path = Path(args.output_dir) / "training_time.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
