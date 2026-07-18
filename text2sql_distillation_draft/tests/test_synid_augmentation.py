import sqlite3

import pytest

from src.synid_sql.augmentation.bird_records import load_bird_train_records
from src.synid_sql.augmentation.jaccard import sql_token_jaccard
from src.synid_sql.augmentation.sql_extract import extract_sql
from src.synid_sql.augmentation.validator import validate_candidate


def test_extract_sql_from_json_and_fenced_blocks():
    assert extract_sql('{"sql": "SELECT 1"}') == "SELECT 1"
    assert extract_sql('```json\n{"sql": "SELECT 2"}\n```') == "SELECT 2"
    assert extract_sql("SELECT 3") == "SELECT 3"


def test_sql_token_jaccard_ignores_case_and_spacing():
    score = sql_token_jaccard("SELECT name FROM singer", "select name from singer")

    assert score == pytest.approx(1.0)


def test_validate_candidate_accepts_execution_equivalent_sql(tmp_path, monkeypatch):
    monkeypatch.setattr("src.synid_sql.augmentation.validator.spider_exec_match", lambda **_: True)
    db_dir = tmp_path / "database" / "toy"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "toy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users(id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO users VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()

    ok, row = validate_candidate(
        {
            "id": 0,
            "db_id": "toy",
            "gold_sql": "SELECT name FROM users WHERE id = 1",
            "candidate_sql": "SELECT users.name FROM users WHERE users.id = 1",
        },
        db_root=tmp_path / "database",
        gamma=1.0,
        timeout_s=5.0,
    )

    assert ok
    assert row["status"] == "accepted"
    assert row["aug_sql"] == "SELECT users.name FROM users WHERE users.id = 1"


def test_validate_candidate_rejects_execution_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("src.synid_sql.augmentation.validator.spider_exec_match", lambda **_: False)
    db_dir = tmp_path / "database" / "toy"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "toy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users(id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO users VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()

    ok, row = validate_candidate(
        {
            "id": 0,
            "db_id": "toy",
            "gold_sql": "SELECT name FROM users WHERE id = 1",
            "candidate_sql": "SELECT name FROM users WHERE id = 2",
        },
        db_root=tmp_path / "database",
        gamma=0.6,
        timeout_s=5.0,
    )

    assert not ok
    assert row["reason"] == "execution_mismatch"


def test_validate_candidate_uses_bird_execution_matcher(tmp_path, monkeypatch):
    monkeypatch.setattr("src.synid_sql.augmentation.validator.bird_exec_match", lambda **_: True)
    db_dir = tmp_path / "database" / "toy"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "toy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users(id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO users VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()

    ok, row = validate_candidate(
        {
            "id": 0,
            "db_id": "toy",
            "gold_sql": "SELECT name FROM users WHERE id = 1",
            "candidate_sql": "SELECT users.name FROM users WHERE users.id = 1",
        },
        benchmark="bird",
        db_root=tmp_path / "database",
        gamma=1.0,
        timeout_s=5.0,
    )

    assert ok
    assert row["status"] == "accepted"


def test_load_bird_train_records_includes_evidence(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "train.json").write_text(
        '[{"db_id":"toy","question":"Q?","evidence":"hint","SQL":"SELECT 1"}]',
        encoding="utf-8",
    )
    (train_dir / "train_tables.json").write_text(
        """
        [
          {
            "db_id": "toy",
            "table_names_original": ["users"],
            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
            "foreign_keys": []
          }
        ]
        """,
        encoding="utf-8",
    )

    records = load_bird_train_records(tmp_path)

    assert records == [
        {
            "id": 0,
            "benchmark": "bird",
            "source_split": "train",
            "db_id": "toy",
            "question": "Q?",
            "evidence": "hint",
            "gold_sql": "SELECT 1",
            "schema": "Tables:\n- users(id, name)",
        }
    ]
