import errno
from typing import Any

import pytest
from pydantic import ValidationError

from gradeflow_core import BUILTINS
from gradeflow_core.common.hints import hint_of
from gradeflow_core.methods.code import code_test
from gradeflow_core.methods.code import runner as code_runner

from .helpers import CODE, code_tests, describe, mark, mark_all, proportions

F = {"expression": "f()", "expected": "1"}


@pytest.mark.parametrize(("returned", "proportion"), [("1", 1), ("2", 0)])
def test_a_test_compares_the_expression_with_the_expected_value(
    returned: str, proportion: float
) -> None:
    assert mark(code_tests(F), f"def f():\n    return {returned}", CODE).proportion == proportion


@pytest.mark.parametrize(
    ("answer", "feedback"),
    [
        (
            "def f():\n    return undefined_variable",
            "f() failed: NameError: name 'undefined_variable' is not defined. Expected 1.",
        ),
        ("oops(", "f() failed: SyntaxError: '(' was never closed. Expected 1."),
        (
            "def f():\n    while True:\n        pass",
            "f() failed: it took longer than 1 second. Expected 1.",
        ),
    ],
)
def test_a_test_that_fails_says_why_and_is_still_marked(answer: str, feedback: str) -> None:
    result = mark(code_tests(F, setup={"time_limit": 1}), answer, CODE)
    assert (result.proportion, result.marked, result.feedback) == (0, True, feedback)


def test_feedback_shows_values_as_python_writes_them() -> None:
    check = code_tests(F, {"expression": "g()", "expected": "'b'"}, combine="proportional")
    assert mark(check, "def f():\n    return 1\ndef g():\n    return 'a'", CODE).feedback == (
        "f() gave 1, as expected.\ng() gave 'a'. Expected 'b'."
    )


def test_code_that_cannot_start_is_left_for_a_person(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

    monkeypatch.setattr(code_runner, "run", unavailable)
    result = mark(code_tests(F), "def f():\n    return 1", CODE)
    assert (result.marked, result.feedback) == (False, "Couldn't mark this answer.")


@pytest.mark.parametrize(
    ("combine", "proportion"), [("all", 0), ("any", 1), ("total", 1 / 6), ("proportional", 0.5)]
)
def test_combine_decides_the_proportion(combine: str, proportion: float) -> None:
    two = code_tests(F, {**F, "expected": "2"}, combine=combine)
    assert (
        mark(two, "def f():\n    return 1", CODE.model_copy(update={"max_marks": 6})).proportion
        == proportion
    )


def test_a_value_json_cannot_carry_still_compares() -> None:
    assert mark(
        code_tests({"expression": "f()", "expected": "range(3)"}),
        "def f():\n    return range(3)",
        CODE,
    ).correct


def test_expected_values_fall_back_to_as_written() -> None:
    assert code_test._worked_out("undefined_name", code_test.TestSetup()) == "undefined_name"


def test_expected_values_are_worked_out_inside_indented_setup_code() -> None:
    setup = code_test.TestSetup(code_before="def f():", code_after="TWO = 2", indent=4)
    assert code_test._worked_out("TWO * 2", setup) == "4"


@pytest.mark.parametrize(
    "setup", [{"indent": -1}, {"time_limit": 0}, {"time_limit": code_test.MAX_TIME_LIMIT_S + 1}]
)
def test_setup_is_bounded(setup: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        code_test.TestSetup.model_validate(setup)


def _expected_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The code run to work out expected values, from now on."""
    code_test._worked_out.cache_clear()
    runs: list[str] = []
    real = code_runner.run

    def counted(code: str, *args: Any, **kwargs: Any) -> None:
        if "output =" not in code:
            runs.append(code.rsplit("expected = ", 1)[-1])
        real(code, *args, **kwargs)

    monkeypatch.setattr(code_runner, "run", counted)
    return runs


def test_expected_values_are_worked_out_once_per_test(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _expected_runs(monkeypatch)
    check = code_tests(
        {"expression": "f()", "expected": "1 + 1"}, {"expression": "f() * 2", "expected": "2 * 2"}
    )
    results = mark_all(check, [f"def f():\n    return {n}" for n in (2, 3, 4)], CODE)
    assert runs == ["1 + 1", "2 * 2"]
    assert proportions(results) == [1, 0, 0]
    assert results[1].feedback == "f() gave 3. Expected 2.\nf() * 2 gave 6. Expected 4."


def test_expected_values_shown_as_written_are_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _expected_runs(monkeypatch)
    check = code_tests({"expression": "f()", "expected": "1 + 1"}, evaluate_expected=False)
    assert mark(check, "def f():\n    return 3", CODE).feedback == "f() gave 3. Expected 1 + 1."
    assert runs == []


def test_description_lists_each_test() -> None:
    tests = (F, {"expression": "g()", "expected": "2"})
    assert describe(code_tests(*tests)) == "Gives `1` for `f()` and `2` for `g()`."
    assert describe(code_tests(*tests, combine="proportional")) == (
        "Gives `1` for `f()` and `2` for `g()`, each counting."
    )


ADD = {"expression": "add(2, 3)", "expected": "5"}
SLIPS = "def add(a, b)\n    retrun a + b"


def test_small_errors_fixed_earn_the_marks_less_the_penalty_and_say_what_was_fixed() -> None:
    result = mark(code_tests(ADD, fix_small_errors=True, fix_penalty=0.25), SLIPS, CODE)
    assert result.proportion == 0.75
    assert result.feedback == (
        "Fixed 2 small errors before testing, for 25% off. "
        "Line 1, added “:” after “def add(a, b)”. Line 2, “retrun” read as “return”.\n"
        "add(2, 3) gave 5, as expected."
    )


def test_fixes_cost_nothing_unless_a_penalty_is_set() -> None:
    result = mark(code_tests(ADD, fix_small_errors=True), SLIPS, CODE)
    assert result.proportion == 1
    assert result.feedback.startswith("Fixed 2 small errors before testing. Line 1, added")


def test_code_is_fixed_only_when_asked_and_within_the_limit() -> None:
    assert mark(code_tests(ADD), SLIPS, CODE).proportion == 0
    assert mark(code_tests(ADD, fix_small_errors=True, max_fixes=1), SLIPS, CODE).proportion == 0
    half = mark(code_tests(ADD, fix_small_errors=True, fix_penalty=0.5), SLIPS, CODE)
    assert half.proportion == 0.5


def test_code_that_passes_as_written_keeps_its_marks_and_feedback() -> None:
    result = mark(code_tests(ADD, fix_small_errors=True), "def add(a, b):\n    return a + b", CODE)
    assert (result.proportion, result.feedback) == (1, "add(2, 3) gave 5, as expected.")


def test_names_the_setup_defines_count_as_defined() -> None:
    setup = {"code_before": "import math\ndef double(x):\n    return 2 * x"}
    check = code_tests({"expression": "f(2)", "expected": "4"}, fix_small_errors=True, setup=setup)
    result = mark(check, "def f(x):\n    return doubel(x)", CODE)
    assert result.proportion == 1
    assert "“doubel” read as “double”" in result.feedback


def test_the_fix_settings_apply_only_when_fixing() -> None:
    [method] = [method for method in BUILTINS if method.name == "gradeflow/code-tests"]
    fields = method.settings.model_fields
    for name in ("fix_penalty", "max_fixes"):
        assert hint_of(fields[name]).applies_when == {"fix_small_errors": True}, name
    assert not hint_of(fields["fix_penalty"]).advanced
    assert fields["fix_penalty"].default == 0
