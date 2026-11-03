import errno
from typing import Any

import pytest
from pydantic import ValidationError

from gradeflow_core.methods.code import code_test
from gradeflow_core.methods.code import runner as code_runner

from .helpers import CODE, code_tests, describe, mark, mark_all, proportions

F = {"expression": "f()", "expected": "1"}
G = {"expression": "g()", "expected": "2"}


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
        ("oops(", "Couldn't run the code: SyntaxError: '(' was never closed (line 1)."),
        (
            "def f():\n    while True:\n        pass",
            "f() failed: it took longer than 1 second. Expected 1.",
        ),
    ],
)
def test_a_test_that_fails_says_why_and_is_still_marked(answer: str, feedback: str) -> None:
    result = mark(code_tests(F, setup={"time_limit": 1}), answer, CODE)
    assert (result.proportion, result.marked, result.feedback) == (0, True, feedback)


def test_feedback_counts_the_tests_passed_and_says_why_the_others_failed() -> None:
    three = [
        F,
        {"expression": "f() + 1", "expected": "3"},
        {"expression": "f() * 3", "expected": "3"},
    ]
    assert mark(code_tests(*three), "def f():\n    return 1", CODE).feedback == (
        "Passed 2 of 3 tests.\nf() + 1 gave 2. Expected 3."
    )
    assert mark(code_tests(*three, combine="any"), "def f():\n    return 2", CODE).feedback == (
        "Passed 1 of 3 tests."
    ), "with Any, failures cost nothing"
    fixed = code_tests(*three[::2], fixes={"colons": True, "colons_penalty": 0.5})
    feedback = mark(fixed, "def f()\n    return 1", CODE).feedback
    assert feedback.endswith("after “def f()” (-50%).\nPassed both tests.")


def test_an_error_that_stops_several_tests_is_said_once() -> None:
    first = code_tests(
        {"expression": "first([5])", "expected": "5"},
        {"expression": "first([])", "expected": "None"},
        {"expression": "first(list())", "expected": "None"},
    )
    assert mark(first, "def first(xs):\n    return xs[0]", CODE).feedback == (
        "Passed 1 of 3 tests.\n"
        "first([]) and first(list()) failed: IndexError: list index out of range."
    )
    assert mark(first, "def first(xs):\n    return xs[1]", CODE).feedback == (
        "Passed 0 of 3 tests.\nAll 3 failed: IndexError: list index out of range."
    )


def test_feedback_shows_values_as_python_writes_them() -> None:
    check = code_tests(F, {"expression": "g()", "expected": "'b'"}, combine="proportional")
    assert mark(check, "def f():\n    return 1\ndef g():\n    return 'a'", CODE).feedback == (
        "Passed 1 of 2 tests.\ng() gave 'a'. Expected 'b'."
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


def test_a_test_runs_within_its_memory_limit() -> None:
    answer = "def f():\n    data = b'x' * (96 * 1024**2)\n    return 1"
    assert mark(code_tests(F, setup={"memory_limit": 256}), answer, CODE).proportion == 1
    result = mark(code_tests(F, setup={"memory_limit": 64}), answer, CODE)
    assert (result.proportion, result.feedback) == (0, "f() failed: MemoryError. Expected 1.")


@pytest.mark.parametrize(
    "setup",
    [
        {"indent": -1},
        {"time_limit": 0},
        {"time_limit": code_runner.MAX_TIME_LIMIT_S + 1},
        {"memory_limit": code_runner.MIN_MEMORY_LIMIT_MB - 1},
        {"memory_limit": code_runner.MAX_MEMORY_LIMIT_MB + 1},
    ],
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
    assert results[1].feedback == (
        "Passed 0 of 2 tests.\nf() gave 3. Expected 2.\nf() * 2 gave 6. Expected 4."
    )


def test_expected_values_shown_as_written_are_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _expected_runs(monkeypatch)
    check = code_tests({"expression": "f()", "expected": "1 + 1"}, evaluate_expected=False)
    assert mark(check, "def f():\n    return 3", CODE).feedback == "f() gave 3. Expected 1 + 1."
    assert runs == []


def test_one_test_is_described_in_a_line() -> None:
    assert describe(code_tests(F)) == "Makes `f()` give `1`."
    assert describe(code_tests(F, combine="none")) == "Doesn't make `f()` give `1`."


@pytest.mark.parametrize(
    ("combine", "opening"),
    [
        ("all", "Passes both tests:"),
        ("any", "Passes any one of these tests:"),
        ("none", "Passes none of these tests:"),
        ("proportional", "Passes these tests, each counting:"),
        ("total", "Passes these tests, marks added up:"),
    ],
)
def test_several_tests_are_listed_under_how_they_count(combine: str, opening: str) -> None:
    assert describe(code_tests(F, G, combine=combine)) == (
        f"{opening}\n- `f()` gives `1`.\n- `g()` gives `2`."
    )
    assert describe(code_tests(F, G, F)).startswith("Passes all 3 tests:\n")


def test_tests_are_listed_with_their_weights() -> None:
    weighted = {**G, "weight": 2}
    assert describe(code_tests(F, weighted, combine="total")) == (
        "Passes these tests, marks added up:\n- `f()` gives `1`.\n- **Weight 2:** `g()` gives `2`."
    )
    assert describe(code_tests(F, {**G, "weight": 0.5}, combine="any")) == (
        "Passes any one of these tests:\n"
        "- `f()` gives `1`.\n- **50% of the marks:** `g()` gives `2`."
    )
    assert describe(code_tests(weighted, combine="total")) == "**Weight 2:** Makes `g()` give `2`."


def test_tests_are_kept_without_surrounding_spaces() -> None:
    spaced = code_tests({"expression": " f() ", "expected": "1\n"})
    assert describe(spaced) == "Makes `f()` give `1`."
    assert mark(spaced, "def f():\n    return 1", CODE).feedback == "f() gave 1, as expected."


def test_tests_found_where_they_are_not_allowed_are_listed_in_the_feedback() -> None:
    check = {"uses": "gradeflow/not-allowed@1", "with": {"check": code_tests(F, G)}}
    assert mark(check, "def f():\n    return 1\ndef g():\n    return 2", CODE).feedback == (
        "Not allowed. Passes both tests:\n- “f()” gives “1”.\n- “g()” gives “2”."
    )


ADD = {"expression": "add(2, 3)", "expected": "5"}
SLIPS = "def add(a, b)\n    retrun a + b"
SLIP_FIXES = {"colons": True, "keywords": True}


def test_each_fix_is_listed_with_its_penalty() -> None:
    fixes = {**SLIP_FIXES, "colons_penalty": 0.25, "keywords_penalty": 0.5}
    result = mark(code_tests(ADD, fixes=fixes), SLIPS, CODE)
    assert result.proportion == 0.25
    assert result.feedback == (
        "Fixed before marking:\n"
        "- Line 1, added “:” after “def add(a, b)” (-25%).\n"
        "- Line 2, “retrun” read as “return” (-50%).\n"
        "add(2, 3) gave 5, as expected."
    )


def test_fixes_cost_nothing_unless_a_penalty_is_set() -> None:
    result = mark(code_tests(ADD, fixes=SLIP_FIXES), SLIPS, CODE)
    assert result.proportion == 1
    assert result.feedback.startswith(
        "Fixed before marking:\n- Line 1, added “:” after “def add(a, b)”.\n"
    )


def test_code_is_fixed_only_as_asked() -> None:
    assert mark(code_tests(ADD), SLIPS, CODE).proportion == 0
    assert mark(code_tests(ADD, fixes={"colons": True}), SLIPS, CODE).proportion == 0


def test_code_that_passes_as_written_keeps_its_marks_and_feedback() -> None:
    result = mark(code_tests(ADD, fixes=SLIP_FIXES), "def add(a, b):\n    return a + b", CODE)
    assert (result.proportion, result.feedback) == (1, "add(2, 3) gave 5, as expected.")


def test_names_the_setup_defines_count_as_defined() -> None:
    setup = {"code_before": "import math\ndef double(x):\n    return 2 * x"}
    check = code_tests({"expression": "f(2)", "expected": "4"}, fixes={"names": True}, setup=setup)
    result = mark(check, "def f(x):\n    return doubel(x)", CODE)
    assert result.proportion == 1
    assert "“doubel” read as “double”" in result.feedback


def test_flat_code_gets_the_marks_of_its_best_layout() -> None:
    flat = "def count(xs):\nn = 0\nfor x in xs:\nif x:\nn += 1\nreturn n"
    tests = [{"expression": "count([1, 0, 1])", "expected": "2"}]
    assert mark(code_tests(*tests), flat, CODE).proportion == 0
    fixes = {"indentation": True, "indentation_penalty": 0.1}
    result = mark(code_tests(*tests, fixes=fixes), flat, CODE)
    assert result.proportion == 0.9
    assert result.feedback.startswith("Fixed before marking:\n- Rebuilt the indentation (-10%).\n")
