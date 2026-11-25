import pytest

from gradeflow_core.editing import CheckRow, Condition, ConditionSubject, SettingRow
from gradeflow_core.steps.conditions import check_subject, condition_template, read_condition


def test_rows_make_one_condition_that_reads_back() -> None:
    rows = (
        CheckRow("test", "correct"),
        CheckRow("unit", "incorrect"),
        CheckRow("q5", "half_correct"),
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


def test_setting_rows_test_a_yes_no_setting_or_a_choice() -> None:
    rows = (
        SettingRow("strict", True),
        SettingRow("units", False),
        SettingRow("mode", "it's"),
        SettingRow("places", 2),
        SettingRow("step", 0.5),
    )
    written = condition_template(Condition(rows, "any"))
    assert written == (
        '{{ settings.strict or not settings.units or settings.mode == "it\'s" '
        "or settings.places == 2 or settings.step == 0.5 }}"
    )
    assert read_condition(written) == Condition(rows, "any")


@pytest.mark.parametrize(
    "written",
    [
        "{{ steps.a.output.proportion > 0.2 }}",
        "{{ x }}",
        "plain",
        "{{ steps.a.output.correct and steps.b.output.correct or steps.c.output.correct }}",
        "{{ steps.a.output.correct and (",
        "{{ results.a.correct }}",
        "{{ settings.mode == true }}",
        "{{ settings.a.b }}",
    ],
)
def test_conditions_other_than_rows_are_text_only(written: str) -> None:
    assert read_condition(written) is None


def test_a_condition_reads_as_words_with_each_row_as_its_subject_says_it() -> None:
    strict = ConditionSubject("Strict", {SettingRow("strict", True): "is on"})
    subjects = [check_subject("test", "Test"), check_subject("units", "Units"), strict]
    rows = (CheckRow("test", "correct"), CheckRow("units", "incorrect"))
    assert Condition(rows).describe(subjects) == (
        "Only when Test is correct and Units isn't correct"
    )
    assert Condition((*rows[:1], SettingRow("strict", True)), "any").describe(subjects) == (
        "Only when Test is correct or Strict is on"
    )
    assert Condition((SettingRow("strict", False),)).describe(subjects) is None
