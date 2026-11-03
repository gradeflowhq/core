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
        ("share_each", [1, 0.5, 0.5, 0]),
    ],
)
def test_each_way_of_comparing(compare: str, expected: list[float]) -> None:
    check = {
        "uses": "gradeflow/multiple-choice@1",
        "with": {"correct": ["a", "b"], "compare": compare},
    }
    assert proportions(mark_all(check, ["a, b", "a", "a, b, c", "c, d"], CHOICE)) == expected


def test_wrong_options_are_a_penalty_with_a_share_for_each_never_below_zero() -> None:
    check = {
        "uses": "gradeflow/multiple-choice@1",
        "with": {"correct": ["a", "b"], "compare": "share_each"},
    }
    assert proportions(mark_all(check, ["a, c", "c, d"], CHOICE)) == [0, 0]


def test_feedback_names_the_correct_options() -> None:
    def feedback(correct: list[str], compare: str, answer: str) -> str:
        check = {
            "uses": "gradeflow/multiple-choice@1",
            "with": {"correct": correct, "compare": compare},
        }
        return mark(check, answer, CHOICE).feedback

    assert feedback(["a", "b"], "exactly", "a, b") == "Correct."
    assert feedback(["a", "b"], "exactly", "c") == "Expected “a” and “b”."
    assert feedback(["a", "b"], "share_each", "a") == "Expected “a” and “b”."
    assert feedback(["a", "b"], "any", "c") == "Expected “a” or “b”."
    assert feedback(["c", "d"], "none", "d") == "Expected none of “c” and “d”."


def test_it_describes_itself_and_refuses_options_the_question_lacks() -> None:
    check = {
        "uses": "gradeflow/multiple-choice@1",
        "with": {"correct": ["a", "z"], "compare": "share_each"},
    }
    assert describe(check) == "Chooses `a` and `z`, a share for each."
    assert describe({**check, "with": {"correct": ["b"]}}) == "Chooses exactly `b`."
    assert describe({**check, "with": {"correct": ["a", "b"], "compare": "all"}}) == (
        "Chooses `a` and `b`, other options allowed."
    )
    assert describe({**check, "with": {"correct": ["a", "b"], "compare": "none"}}) == (
        "Chooses none of `a` and `b`."
    )
    assert describe({**check, "with": {"correct": ["a", "b"], "compare": "any"}}) == (
        "Chooses `a` or `b`."
    )
    assert problems_of(check, CHOICE) == [("", ("correct", 1), "z isn't one of the options.")]


@pytest.mark.parametrize(
    ("compare", "first_line"),
    [
        ("exactly", "Chooses exactly these:"),
        ("all", "Chooses all of these, other options allowed:"),
        ("none", "Chooses none of these:"),
        ("any", "Chooses any of these:"),
        ("share_each", "Chooses these, a share for each:"),
    ],
)
def test_long_options_are_listed_one_per_line(compare: str, first_line: str) -> None:
    options = [
        "The mitochondria release energy from glucose",
        "The ribosomes make proteins from amino acids",
    ]
    check = {
        "uses": "gradeflow/multiple-choice@1",
        "with": {"correct": options, "compare": compare},
    }
    assert describe(check) == f"{first_line}\n- `{options[0]}`\n- `{options[1]}`"


def test_the_correct_options_are_read_as_the_question_reads_its_own() -> None:
    check = {"uses": "gradeflow/multiple-choice@1", "with": {"correct": [" B", "C"]}}
    assert proportions(mark_all(check, ["b, c", "B, C", "b"], CHOICE)) == [1, 1, 0]
    assert problems_of(check, CHOICE) == []
    case_sensitive = ChoiceQuestion.model_validate(
        {"options": ["A", "b"], "reading": {"ignore_case": False}}
    )
    assert problems_of({**check, "with": {"correct": ["a"]}}, case_sensitive) == [
        ("", ("correct", 0), "a isn't one of the options.")
    ]


def test_its_settings_schema_offers_the_questions_options() -> None:
    multiple_choice = BUILTINS.find("gradeflow/multiple-choice@1")
    schema = multiple_choice.settings_schema(Position(QuestionSet({"Q1": CHOICE}), "Q1"))
    assert schema["properties"]["correct"]["items"]["enum"] == ["a", "b", "c", "d"]
