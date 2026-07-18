from types import SimpleNamespace
import sys
import types

import numpy as np

sys.modules.setdefault("deepspeed", types.ModuleType("deepspeed"))
accelerate_stub = types.ModuleType("accelerate")
accelerate_stub.load_checkpoint_and_dispatch = None
sys.modules.setdefault("accelerate", accelerate_stub)

peft_stub = types.ModuleType("peft")
peft_stub.get_peft_model = None
peft_stub.LoraConfig = object
peft_stub.TaskType = object
peft_stub.PeftModel = object
sys.modules.setdefault("peft", peft_stub)

from data_utils.lm_datasets import LMTrainDataset


def test_teacher_collate_uses_t_max_length_instead_of_student_max_length():
    dataset = LMTrainDataset.__new__(LMTrainDataset)
    dataset.args = SimpleNamespace(model_type="qwen", type="synid", t_max_length=7)
    dataset.pad_id = 0
    dataset.max_length = 5
    dataset.max_prompt_length = 4

    sample = {
        "input_ids": np.array([101, 102, -1, 201, 202, 203, 204, 205]),
        "t_input_ids": np.array([101, 102, -1, 201, 202, 203, 204, 205]),
    }

    model_data, _, _, t_model_data, _ = dataset.collate([sample])

    assert int(model_data["attention_mask"].sum().item()) == 4
    assert int(t_model_data["attention_mask"].sum().item()) == 6
