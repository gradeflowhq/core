import pytest

from gradeflow_core.editing import Condition, ConditionRow
from gradeflow_core.editing.conditions import read_condition, written_condition


def test_rows_make_one_condition_that_reads_back() -> None:
    rows = (
        ConditionRow("test", "correct"),
        ConditionRow("unit", "incorrect"),
        ConditionRow("q5", "half"),
    )
    written = written_condition(Condition(rows))
    assert written == (
        "{{ checks.test.correct and not checks.unit.correct and checks.q5.proportion >= 0.5 }}"
    )
    assert read_condition(written) == Condition(rows)
    assert read_condition(written_condition(Condition(rows[:2], "any"))) == Condition(
        rows[:2], "any"
    )
    assert written_condition(Condition((ConditionRow("", "correct"),))) == "", "always"


@pytest.mark.parametrize(
    "written",
    [
        "{{ checks.a.proportion > 0.2 }}",
        "{{ x }}",
        "plain",
        "{{ checks.a.correct and checks.b.correct or checks.c.correct }}",
        "{{ checks.a.correct and (",
        "{{ results.a.correct }}",
    ],
)
def test_conditions_other_than_rows_are_text_only(written: str) -> None:
    assert read_condition(written) is None


def test_a_condition_reads_as_words_with_each_test_as_it_reads() -> None:
    rows = (ConditionRow("Test", "correct"), ConditionRow("Units", "incorrect"))
    assert Condition(rows).describe() == "Only when Test is correct and Units isn't correct"
    assert Condition(rows, "any").describe(str.lower) == (
        "Only when test is correct or units isn't correct"
    )
    assert list(Condition.tests) == ["correct", "incorrect", "half"]
