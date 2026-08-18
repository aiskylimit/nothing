from __future__ import annotations

import json
from types import SimpleNamespace

from pvsd.common.local_logging import LocalJSONLCallback


def test_local_jsonl_callback_writes_metrics(tmp_path):
    callback = LocalJSONLCallback()
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(is_world_process_zero=True, global_step=0)
    callback.on_train_begin(args, state, None)

    state.global_step = 2
    callback.on_log(args, state, None, logs={"loss": 1.25, "learning_rate": 5e-6})

    record = json.loads((tmp_path / "training_log.jsonl").read_text(encoding="utf-8"))
    assert record["step"] == 2
    assert record["loss"] == 1.25
    assert record["learning_rate"] == 5e-6
    assert record["elapsed_seconds"] >= 0


def test_local_jsonl_callback_only_writes_on_main_process(tmp_path):
    callback = LocalJSONLCallback()
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(is_world_process_zero=False, global_step=2)

    callback.on_log(args, state, None, logs={"loss": 1.25})

    assert not (tmp_path / "training_log.jsonl").exists()
