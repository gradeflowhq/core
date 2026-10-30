from typing import Any

import pytest

from gradeflow_core import InvalidError
from gradeflow_core.questions.reading import ListReading
from gradeflow_core.questions.values import ValuesQuestion

from .helpers import describe, mark, mark_all, problems_of, proportions

FOUR = ValuesQuestion(value_types=["text"] * 4)
ANSWERS = ["a, b, c, d", "a, c, b, d", "d, a, b, c", "d, c, b, a"]


def order(*items: str, **settings: Any) -> dict[str, Any]:
    return {"uses": "gradeflow/order@1", "with": {"order": list(items), **settings}}


@pytest.mark.parametrize(
    ("marks_for", "expected"),
    [
        ("exact", [1, 0, 0, 0]),
        ("longest", [1, 0.75, 0.75, 0.25]),
        ("pairs", [1, 0, 0.6667, 0]),
    ],
)
def test_each_way_of_giving_marks(marks_for: str, expected: list[float]) -> None:
    check = order("a", "b", "c", "d", marks_for=marks_for)
    assert proportions(mark_all(check, ANSWERS, FOUR)) == expected


def test_a_repeated_pair_counts_once() -> None:
    check = order("a", "b", "c", "d", marks_for="pairs")
    assert mark(check, "a, b, a, b", FOUR).proportion == pytest.approx(1 / 3)


def test_feedback_says_how_much_is_in_order_and_the_right_order() -> None:
    expected = "Expected a, b, c, d."
    assert mark(order("a", "b", "c", "d"), "a, b, c, d", FOUR).feedback == "Correct."
    assert mark(order("a", "b", "c", "d"), "a, c, b, d", FOUR).feedback == (
        f"3 of 4 in order. {expected}"
    )
    pairs = order("a", "b", "c", "d", marks_for="pairs")
    assert mark(pairs, "d, a, b, c", FOUR).feedback == f"2 of 3 pairs in order. {expected}"
    exact = order("a", "b", "c", "d", marks_for="exact")
    assert mark(exact, "d, c, b, a", FOUR).feedback == expected


def test_items_are_read_as_the_question_reads_values() -> None:
    question = ValuesQuestion(
        value_types=["text"] * 3, reading=ListReading(ignore_case=True, separator=";")
    )
    check = order("Alpha", "Beta", "Gamma", marks_for="exact")
    assert mark(check, "alpha; BETA; gamma", question).proportion == 1
    numbers = ValuesQuestion(value_types=["number"] * 3)
    assert mark(order("1", "2.5", "3/1"), "1, 2.5, 3", numbers).proportion == 1


def test_it_describes_the_order() -> None:
    assert describe(order("prophase", "metaphase", "anaphase", marks_for="exact")) == (
        "Puts `prophase`, `metaphase`, `anaphase` in order."
    )
    assert describe(order("a", "b")) == "Puts `a`, `b` in order, with a share for items in order."
    assert describe(order("a", "b", marks_for="pairs")) == (
        "Puts `a`, `b` in order, with a share for neighbours in order."
    )


def test_the_question_needs_a_value_for_each_item_all_of_one_type() -> None:
    three = ValuesQuestion(value_types=["text"] * 3)
    assert problems_of(order("a", "b"), three) == [
        ("", ("order",), "List 3 items, one for each value.")
    ]
    mixed = ValuesQuestion(value_types=["text", "number"])
    assert problems_of(order("a", "1"), mixed) == [
        ("", ("order",), "The question's values must all be text or all numbers.")
    ]


def test_each_item_must_read() -> None:
    numbers = ValuesQuestion(value_types=["number"] * 3)
    assert problems_of(order("1", "two", "N/A"), numbers) == [
        ("", ("order", 1), "Couldn't read “two” as a number."),
        ("", ("order", 2), "It reads as a blank answer."),
    ]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (["a"], "Add at least 2."),
        (["a", "a"], "Each item must be different."),
        (["a", " "], "Can't be blank."),
    ],
)
def test_items_are_at_least_two_different_ones(items: list[str], message: str) -> None:
    with pytest.raises(InvalidError, match=message):
        problems_of(order(*items), FOUR)
