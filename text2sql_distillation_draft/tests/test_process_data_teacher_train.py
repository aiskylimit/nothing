import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from process_data import Encoder, get_builder_dtype, merge_teacher_train_data


def _json_line(record):
    return json.dumps(record, ensure_ascii=False) + "\n"


def test_merge_teacher_train_data_adds_teacher_prompt_fields(tmp_path: Path):
    teacher_path = tmp_path / "teacher_train.jsonl"
    teacher_path.write_text(
        _json_line(
            {
                "t_system_prompt": "teacher system",
                "t_user_prompt": "teacher user",
                "response": '{"sql": "SELECT 1"}',
            }
        ),
        encoding="utf-8",
    )
    student_lines = [
        _json_line(
            {
                "system_prompt": "student system",
                "user_prompt": "student user",
                "response": '{"sql": "SELECT 1"}',
            }
        )
    ]

    merged_lines = merge_teacher_train_data(student_lines, str(teacher_path))

    merged = json.loads(merged_lines[0])
    assert merged["system_prompt"] == "student system"
    assert merged["user_prompt"] == "student user"
    assert merged["t_system_prompt"] == "teacher system"
    assert merged["t_user_prompt"] == "teacher user"
    assert merged["response"] == '{"sql": "SELECT 1"}'


def test_merge_teacher_train_data_rejects_response_mismatch(tmp_path: Path):
    teacher_path = tmp_path / "teacher_train.jsonl"
    teacher_path.write_text(
        _json_line(
            {
                "t_system_prompt": "teacher system",
                "t_user_prompt": "teacher user",
                "response": '{"sql": "SELECT 2"}',
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="responses differ"):
        merge_teacher_train_data(
            [
                _json_line(
                    {
                        "system_prompt": "student system",
                        "user_prompt": "student user",
                        "response": '{"sql": "SELECT 1"}',
                    }
                )
            ],
            str(teacher_path),
        )


def test_merge_teacher_train_data_rejects_missing_teacher_fields(tmp_path: Path):
    teacher_path = tmp_path / "teacher_train.jsonl"
    teacher_path.write_text(
        _json_line(
            {
                "t_system_prompt": "teacher system",
                "response": '{"sql": "SELECT 1"}',
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing teacher fields"):
        merge_teacher_train_data(
            [
                _json_line(
                    {
                        "system_prompt": "student system",
                        "user_prompt": "student user",
                        "response": '{"sql": "SELECT 1"}',
                    }
                )
            ],
            str(teacher_path),
        )


def test_encoder_keeps_only_response_tokens_after_prompt_truncation():
    class FakeTokenizer:
        eos_token_id = 0

        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return "".join(message["content"] for message in messages)

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert not add_special_tokens
            return [ord(character) for character in text]

    args = SimpleNamespace(
        split="train",
        max_prompt_length=3,
        t_max_prompt_length=10,
    )
    encoder = Encoder(args)
    Encoder.tokenizer = FakeTokenizer()
    response = "SQL"

    _, _, prompt_tokens, response_tokens, _, _, _ = encoder.encode(
        json.dumps(
            {
                "system_prompt": "long",
                "user_prompt": "prompt",
                "response": response,
            }
        )
    )

    assert len(prompt_tokens) == args.max_prompt_length
    assert response_tokens == FakeTokenizer.encode(response) + [FakeTokenizer.eos_token_id]


def test_encoder_falls_back_when_tokenizer_rejects_enable_thinking():
    class LlamaLikeTokenizer:
        eos_token_id = 2

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            return "|".join(message["content"] for message in messages)

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert not add_special_tokens
            return [ord(character) for character in text]

    args = SimpleNamespace(
        split="train",
        max_prompt_length=100,
        t_max_prompt_length=100,
    )
    encoder = Encoder(args)
    Encoder.tokenizer = LlamaLikeTokenizer()

    _, prompt, _, response_tokens, _, _, _ = encoder.encode(
        json.dumps(
            {
                "system_prompt": "system",
                "user_prompt": "user",
                "response": "SQL",
            }
        )
    )

    assert prompt == "system|user"
    assert response_tokens == LlamaLikeTokenizer.encode("SQL") + [LlamaLikeTokenizer.eos_token_id]


def test_builder_dtype_uses_int32_for_large_vocab_tokenizers():
    assert get_builder_dtype(SimpleNamespace(vocab_size=64000)) == np.uint16
    assert get_builder_dtype(SimpleNamespace(vocab_size=128000)) == np.int32
