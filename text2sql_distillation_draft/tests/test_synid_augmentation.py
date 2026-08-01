import sqlite3

import pytest

from src.synid_sql.augmentation.similarity import sql_sequence_similarity
from src.synid_sql.augmentation.sql_extract import extract_sql
from src.synid_sql.augmentation.validator import validate_candidate


def test_extract_sql_from_json_and_fenced_blocks():
    assert extract_sql('{"sql": "SELECT 1"}') == "SELECT 1"
    assert extract_sql('```json\n{"sql": "SELECT 2"}\n```') == "SELECT 2"
    assert extract_sql("SELECT 3") == "SELECT 3"


def test_sql_sequence_similarity_uses_normalized_sql():
    score = sql_sequence_similarity("SELECT name FROM singer", "select name from singer")

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
    assert row["similarity"] < 1.0


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


def test_validate_candidate_keeps_execution_correct_high_similarity_for_final_selection(tmp_path, monkeypatch):
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
            "candidate_sql": "SELECT name FROM users WHERE id = 1",
        },
        db_root=tmp_path / "database",
        gamma=0.9,
        timeout_s=5.0,
    )

    assert not ok
    assert row["reason"] == "sql_similarity_too_high"
    assert row["similarity"] == pytest.approx(1.0)
    assert row["repairable"]


