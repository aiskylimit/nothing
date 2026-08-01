import json

import pytest

from scripts.format_spider_synid_jsonl import (
    JSON_SCHEMA,
    make_teacher_record,
    validate_student_alignment,
)


def test_teacher_record_contains_only_privileged_prompt_and_response():
    sample = {
        "db_id": "demo",
        "question": "How many users are there?",
        "query": "SELECT COUNT(*) FROM users",
    }
    record = make_teacher_record(
        sample,
        {"demo": "Tables:\n- users(id)"},
        teacher_system_template="teacher system\n{json_schema}",
        teacher_user_template=(
            "Question: {question}\nSchema: {schema}\nReference: {gold_sql}"
        ),
    )

    assert set(record) == {"t_system_prompt", "t_user_prompt", "response"}
    assert JSON_SCHEMA in record["t_system_prompt"]
    assert "SELECT COUNT(*) FROM users" in record["t_user_prompt"]
    assert json.loads(record["response"]) == {
        "sql": "SELECT COUNT(*) FROM users",
    }


def test_teacher_rows_must_align_with_student_responses(tmp_path):
    student_path = tmp_path / "train.jsonl"
    student_path.write_text(
        json.dumps({"response": '{"sql": "SELECT 1"}'}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="responses differ"):
        validate_student_alignment(
            [{"response": '{"sql": "SELECT 2"}'}],
            student_path,
        )
