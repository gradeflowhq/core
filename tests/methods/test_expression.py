from typing import Any

import pytest

from gradeflow_core import InvalidError

from .helpers import describe, mark, mark_all, problems_of, proportions

EXPRESSION = "gradeflow/equivalent-expression@1"


def check(*answers: str) -> dict[str, Any]:
    return {"uses": EXPRESSION, "with": {"answers": list(answers)}}


def test_any_equal_expression_gets_full_marks() -> None:
    answers = ["2X+2", "2 + 2x", "2(x + 1)", "x + x + 2", "2x + 1", "2x^2 + 2", "4x/2 + 2"]
    assert proportions(mark_all(check("2x + 2"), answers)) == [1, 1, 1, 1, 0, 0, 1]


def test_expressions_of_several_variables_and_functions() -> None:
    assert proportions(
        mark_all(
            check("(a + b)(a - b)"),
            ["a² \N{MINUS SIGN} b²", "a^2 - b^2", "(a - b)^2", "(b + a)(a - b)"],
        )
    ) == [1, 1, 0, 1]
    assert proportions(
        mark_all(check("2 sin(x) cos(x)"), ["sin(2x)", "2sin(x)cos(x)", "sin(x)"])
    ) == [
        1,
        1,
        0,
    ]
    assert proportions(mark_all(check("ln(x^2)"), ["2ln(x)", "2log(x)"])) == [1, 0]


def test_one_of_several_accepted_expressions_is_enough() -> None:
    assert proportions(mark_all(check("x + 1", "x - 1"), ["1 + x", "-1 + x", "x"])) == [1, 1, 0]


def test_feedback_says_what_was_expected() -> None:
    assert mark(check("2x + 2"), "2(x+1)").feedback == "Correct."
    assert mark(check("2x + 2"), "2x").feedback == "Expected an expression equal to “2x + 2”."
    assert (
        mark(check("x + 1", "x - 1"), "x").feedback
        == "Expected an expression equal to “x + 1” or “x - 1”."
    )


def test_an_answer_that_isnt_an_expression_gets_no_marks() -> None:
    for answer in ["2x +", "2x = 2", "__import__('os')", "x.real"]:
        result = mark(check("2x + 2"), answer)
        assert (result.proportion, result.feedback) == (
            0,
            "Couldn't read it as an expression. Expected an expression equal to “2x + 2”.",
        )


def test_a_variable_the_accepted_expressions_dont_use_gets_no_marks() -> None:
    result = mark(check("2x + 2"), "2y + 2")
    assert (result.proportion, result.feedback) == (
        0,
        "Uses y. Expected an expression equal to “2x + 2”.",
    )
    assert mark(check("2x + 2"), "2x + 2 + 0z").feedback == (
        "Uses z. Expected an expression equal to “2x + 2”."
    )


def test_answers_it_cant_work_out_are_not_equal() -> None:
    assert proportions(mark_all(check("x"), ["x/(x - x)", "9^9^9 x", "sqrt(-x)"])) == [0, 0, 0]


def test_points_where_the_accepted_expression_cant_be_worked_out_are_skipped() -> None:
    assert proportions(mark_all(check("sqrt(x - 1)"), ["(x - 1)^0.5", "sqrt(x)"])) == [1, 0]


def test_describes_the_accepted_expressions() -> None:
    assert describe(check("2x + 2")) == "Equals `2x + 2`."
    assert describe(check("x + 1", "x - 1")) == "Equals `x + 1` or `x - 1`."


def test_accepted_expressions_must_read_and_be_workable() -> None:
    assert problems_of(check("2x + 2")) == []
    assert problems_of(check("x", "2x +", "sqrt(-x)", "1/0")) == [
        ("", ("answers", 1), "Couldn't read “2x +” as an expression."),
        ("", ("answers", 2), "Can't be worked out for most values of its variables."),
        ("", ("answers", 3), "Can't be worked out for most values of its variables."),
    ]
    with pytest.raises(InvalidError, match="Add at least one"):
        problems_of(check())
