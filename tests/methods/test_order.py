from typing import Any

import pytest

from gradeflow_core import InvalidError
from gradeflow_core.editing import Position, compatible_methods
from gradeflow_core.questions.list import ListQuestion
from gradeflow_core.questions.parts import AnswerPart
from gradeflow_core.questions.question_set import QuestionSet
from gradeflow_core.questions.reading import ListReading

from .helpers import describe, mark, mark_all, problems_of, proportions

TEXT_LIST = ListQuestion()
ANSWERS = ["a, b, c, d", "a, c, b, d", "d, a, b, c", "d, c, b, a"]


def order(*items: str, **settings: Any) -> dict[str, Any]:
    return {"uses": "gradeflow/order@1", "with": {"order": list(items), **settings}}


@pytest.mark.parametrize(
    ("marks_for", "expected"),
    [
        ("exact", [1, 0, 0, 0]),
        ("items_in_order", [1, 0.75, 0.75, 0.25]),
        ("pairs_in_order", [1, 0, 0.6667, 0]),
    ],
)
def test_each_way_of_giving_marks(marks_for: str, expected: list[float]) -> None:
    check = order("a", "b", "c", "d", marks_for=marks_for)
    assert proportions(mark_all(check, ANSWERS, TEXT_LIST)) == expected


def test_a_repeated_pair_counts_once() -> None:
    check = order("a", "b", "c", "d", marks_for="pairs_in_order")
    assert mark(check, "a, b, a, b", TEXT_LIST).proportion == pytest.approx(1 / 3)


def test_feedback_says_how_much_is_in_order_and_the_correct_order() -> None:
    expected = "Expected a, b, c, d."
    assert mark(order("a", "b", "c", "d"), "a, b, c, d", TEXT_LIST).feedback == "Correct."
    assert mark(order("a", "b", "c", "d"), "a, c, b, d", TEXT_LIST).feedback == (
        f"3 of 4 in order. {expected}"
    )
    pairs_in_order = order("a", "b", "c", "d", marks_for="pairs_in_order")
    assert (
        mark(pairs_in_order, "d, a, b, c", TEXT_LIST).feedback
        == f"2 of 3 pairs in order. {expected}"
    )
    exact = order("a", "b", "c", "d", marks_for="exact")
    assert mark(exact, "d, c, b, a", TEXT_LIST).feedback == expected


def test_extra_items_count_against_the_share() -> None:
    check = order("a", "b", "c", "d")
    assert mark(check, "a, b, x, c, d", TEXT_LIST).feedback == (
        "4 of 5 in order. 1 extra item. Expected a, b, c, d."
    )
    assert mark(check, "a, b, x, c, d", TEXT_LIST).proportion == 0.8
    assert mark(check, "a, b, c", TEXT_LIST).proportion == 0.75
    pairs_in_order = order("a", "b", "c", "d", marks_for="pairs_in_order")
    assert mark(pairs_in_order, "a, b, c, d, e", TEXT_LIST).proportion == 0.75


def test_items_are_read_as_the_question_reads_them() -> None:
    question = ListQuestion(reading=ListReading(ignore_case=True, separator=";"))
    check = order("Alpha", "Beta", "Gamma", marks_for="exact")
    assert mark(check, "alpha; BETA; gamma", question).proportion == 1
    numbers = ListQuestion(item="number")
    assert mark(order("1", "2.5", "3/1"), "1, 2.5, 3", numbers).proportion == 1


def test_it_describes_the_order() -> None:
    assert describe(order("prophase", "metaphase", "anaphase", marks_for="exact")) == (
        "Puts `prophase`, `metaphase`, `anaphase` in order."
    )
    assert (
        describe(order("a", "b"))
        == "Puts `a`, `b` in order, with partial marks for items in order."
    )
    assert describe(order("a", "b", marks_for="pairs_in_order")) == (
        "Puts `a`, `b` in order, with partial marks for pairs in order."
    )
    stages = [
        "Interphase G1",
        "S phase",
        "Interphase G2",
        "Prophase",
        "Prometaphase",
        "Metaphase",
        "Anaphase",
    ]
    assert describe(order(*stages, marks_for="exact")) == "Puts these in order:\n" + "\n".join(
        f"{place}. `{stage}`" for place, stage in enumerate(stages, start=1)
    )


def test_it_marks_a_list_of_text_or_numbers() -> None:
    questions = QuestionSet(
        {
            "Q1": TEXT_LIST,
            "Q2": ListQuestion(
                item="parts", parts=[AnswerPart(name="problem"), AnswerPart(name="change")]
            ),
        }
    )
    names = {
        question_id: [
            method.name for method in compatible_methods(Position(questions, question_id))
        ]
        for question_id in questions
    }
    assert "gradeflow/order" in names["Q1"]
    assert "gradeflow/order" not in names["Q2"]


def test_each_item_must_be_readable() -> None:
    numbers = ListQuestion(item="number")
    assert problems_of(order("1", "two", "N/A"), numbers) == [
        ("", "/with/order/1", "Couldn't read “two” as a number."),
        ("", "/with/order/2", "It reads as a blank answer."),
    ]


def test_each_item_must_read_differently() -> None:
    numbers = ListQuestion(item="number")
    assert problems_of(order("1", "1.0", "3"), numbers) == [
        ("", "/with/order/1", "It reads the same as item 1.")
    ]
    assert problems_of(order("A", "b", "a"), TEXT_LIST) == [
        ("", "/with/order/2", "It reads the same as item 1.")
    ]
    case_sensitive = ListQuestion(reading=ListReading(ignore_case=False))
    assert problems_of(order("A", "b", "a"), case_sensitive) == []


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (["a"], "Add at least 2."),
        (["a", " "], "Can't be blank."),
    ],
)
def test_items_are_at_least_two_and_not_blank(items: list[str], message: str) -> None:
    with pytest.raises(InvalidError, match=message):
        problems_of(order(*items), TEXT_LIST)
