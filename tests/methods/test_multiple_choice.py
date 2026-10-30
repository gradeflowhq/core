import pytest

from gradeflow_core import BUILTINS, QuestionSet
from gradeflow_core.editing import Position
from gradeflow_core.questions.choice import ChoiceQuestion

from .helpers import describe, mark, mark_all, problems_of, proportions

CHOICE = ChoiceQuestion(options=frozenset({"a", "b", "c", "d"}), allow_multiple=True)


@pytest.mark.parametrize(
    ("compare", "expected"),
    [
        ("exactly", [1, 0, 0, 0]),
        ("all", [1, 0, 1, 0]),
        ("none", [0, 0, 0, 1]),
        ("any", [1, 1, 1, 0]),
        ("each", [1, 0.5, 0.5, 0]),
    ],
)
def test_each_way_of_comparing(compare: str, expected: list[float]) -> None:
    check = {
        "uses": "gradeflow/multiple-choice@1",
        "with": {"correct": ["a", "b"], "compare": compare},
    }
    assert proportions(mark_all(check, ["a, b", "a", "a, b, c", "c, d"], CHOICE)) == expected


def test_wrong_options_take_marks_off_when_each_one_counts_never_below_zero() -> None:
    check = {
        "uses": "gradeflow/multiple-choice@1",
        "with": {"correct": ["a", "b"], "compare": "each"},
    }
    assert proportions(mark_all(check, ["a, c", "c, d"], CHOICE)) == [0, 0]


def test_feedback_names_the_right_options() -> None:
    def feedback(correct: list[str], compare: str, answer: str) -> str:
        check = {
            "uses": "gradeflow/multiple-choice@1",
            "with": {"correct": correct, "compare": compare},
        }
        return mark(check, answer, CHOICE).feedback

    assert feedback(["a", "b"], "exactly", "a, b") == "Correct."
    assert feedback(["a", "b"], "exactly", "c") == "Expected “a” and “b”."
    assert feedback(["a", "b"], "each", "a") == "Expected “a” and “b”."
    assert feedback(["a", "b"], "any", "c") == "Expected “a” or “b”."
    assert feedback(["c", "d"], "none", "d") == "Expected none of “c” and “d”."


def test_it_describes_itself_and_refuses_options_the_question_lacks() -> None:
    check = {
        "uses": "gradeflow/multiple-choice@1",
        "with": {"correct": ["a", "z"], "compare": "each"},
    }
    assert describe(check) == "Chooses `a` and `z`, each counting."
    assert describe({**check, "with": {"correct": ["b"]}}) == "Chooses exactly `b`."
    assert describe({**check, "with": {"correct": ["a", "b"], "compare": "all"}}) == (
        "Chooses `a` and `b`, extras allowed."
    )
    assert describe({**check, "with": {"correct": ["a", "b"], "compare": "none"}}) == (
        "Chooses none of `a` and `b`."
    )
    assert describe({**check, "with": {"correct": ["a", "b"], "compare": "any"}}) == (
        "Chooses `a` or `b`."
    )
    assert problems_of(check, CHOICE) == [("", ("correct", 1), "z isn't one of the options.")]


def test_the_right_options_are_read_as_the_question_reads_its_own() -> None:
    check = {"uses": "gradeflow/multiple-choice@1", "with": {"correct": [" B", "C"]}}
    assert proportions(mark_all(check, ["b, c", "B, C", "b"], CHOICE)) == [1, 1, 0]
    assert problems_of(check, CHOICE) == []
    exact = ChoiceQuestion.model_validate(
        {"options": ["A", "b"], "reading": {"ignore_case": False}}
    )
    assert problems_of({**check, "with": {"correct": ["a"]}}, exact) == [
        ("", ("correct", 0), "a isn't one of the options.")
    ]


def test_its_settings_schema_offers_the_questions_options() -> None:
    choice = next(method for method in BUILTINS if method.ref == "gradeflow/multiple-choice@1")
    schema = choice.settings_schema(Position(QuestionSet({"Q1": CHOICE}), "Q1"))
    assert schema["properties"]["correct"]["items"]["enum"] == ["a", "b", "c", "d"]
