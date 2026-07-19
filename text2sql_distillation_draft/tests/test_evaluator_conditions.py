import sys
import types
from pathlib import Path


EVALUATOR_DIR = Path(__file__).resolve().parents[1] / "src" / "evaluator"
sys.path.insert(0, str(EVALUATOR_DIR))

process_sql = types.ModuleType("process_sql")
process_sql.get_schema = lambda db: None
process_sql.Schema = object
process_sql.get_sql = lambda schema, sql: None
sys.modules["process_sql"] = process_sql

exec_eval = types.ModuleType("exec_eval")
exec_eval.TIMEOUT = 60
exec_eval.eval_exec_match = lambda **kwargs: False
sys.modules["exec_eval"] = exec_eval

from evaluation import eval_and_or, eval_where


def test_eval_and_or_ignores_malformed_condition_units_in_connector_slots():
    cond_unit = (False, 2, (0, (0, "__all__", False), None), None, None)
    nested_sql_cond_unit = (
        False,
        8,
        (0, (0, "__all__", False), None),
        {"select": (False, [])},
        None,
    )

    pred = {"where": [cond_unit, nested_sql_cond_unit]}
    label = {"where": [cond_unit, "or", cond_unit]}

    assert eval_and_or(pred, label) == (1, 1, 0)


def test_eval_where_marks_malformed_adjacent_condition_units_wrong():
    first_cond = (False, 2, (0, (0, "documents.author_name", False), None), None, None)
    second_cond = (False, 5, (0, (0, "documents.author_name", False), None), None, None)

    pred = {"where": [first_cond, second_cond]}
    label = {"where": [first_cond, "and", second_cond]}

    assert eval_where(pred, label) == (2, 2, 0, 0)
