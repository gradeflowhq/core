import pytest

from gradeflow_core.editing import Condition, ConditionRow
from gradeflow_core.steps.conditions import condition_template, read_condition


def test_rows_make_one_condition_that_reads_back() -> None:
    rows = (
        ConditionRow("test", "correct"),
        ConditionRow("unit", "incorrect"),
        ConditionRow("q5", "half_correct"),
    )
    written = condition_template(Condition(rows))
    assert written == (
        "{{ steps.test.output.correct and not steps.unit.output.correct "
        "and steps.q5.output.proportion >= 0.5 }}"
    )
    assert read_condition(written) == Condition(rows)
    assert read_condition(condition_template(Condition(rows[:2], "any"))) == Condition(
        rows[:2], "any"
    )
    assert condition_template(Condition()) == "", "always"


@pytest.mark.parametrize(
    "written",
    [
        "{{ steps.a.output.proportion > 0.2 }}",
        "{{ x }}",
        "plain",
        "{{ steps.a.output.correct and steps.b.output.correct or steps.c.output.correct }}",
        "{{ steps.a.output.correct and (",
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
    assert list(Condition.test_labels) == ["correct", "incorrect", "half_correct"]
