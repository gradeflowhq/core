import pytest

from gradeflow_core import BUILTINS, InvalidError
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.values import ValuesQuestion

from .helpers import describe, mark, mark_all, problems_of, proportions, scheme


def test_pattern_matches_with_its_options() -> None:
    check = {"uses": "gradeflow/pattern@1", "with": {"pattern": "^colou?r$", "ignore_case": True}}
    assert proportions(mark_all(check, ["colour", "Color", "colours"])) == [1, 1, 0]
    assert mark(check, "color").feedback == "Right format."
    assert mark(check, "red").feedback == "Not in the expected format."
    assert describe(check) == "Matches the pattern `^colou?r$`, ignoring capitals."


def test_a_pattern_must_compile() -> None:
    with pytest.raises(InvalidError, match="Not a valid pattern"):
        problems_of({"uses": "gradeflow/pattern@1", "with": {"pattern": "("}})


def test_a_number_range_needs_its_limits_in_order() -> None:
    with pytest.raises(InvalidError, match="The highest number is below the lowest"):
        problems_of(
            {"uses": "gradeflow/number-range@1", "with": {"min_value": 2, "max_value": 1}},
            NumberQuestion(),
        )


def test_length_counts_words_or_characters() -> None:
    check = {
        "uses": "gradeflow/length@1",
        "with": {"min_length": 3, "max_length": 5, "count": "words"},
    }
    assert proportions(mark_all(check, ["one two three", "one two", "a b c d e f"])) == [1, 0, 0]
    assert mark(check, "one two").feedback == "Too short: 2 words. Expected at least 3."
    assert mark(check, "a b c d e f").feedback == "Too long: 6 words. Expected at most 5."
    assert describe(check) == "Has `3` to `5` words."


def test_a_blank_answer_gets_no_marks_before_the_method_marks_it() -> None:
    check = {"uses": "gradeflow/length@1", "with": {"min_length": 1}}
    assert (mark(check, "").proportion, mark(check, "").feedback) == (0, "No answer.")
    with pytest.raises(InvalidError, match="blank"):
        scheme({"question": "Q1", "blank": "mark", **check})


def test_full_marks_gives_everyone_full_marks_blank_answers_too() -> None:
    check = {"uses": "gradeflow/full-marks@1"}
    assert proportions(mark_all(check, ["anything", "", "N/A"])) == [1, 1, 1]


def test_only_full_marks_marks_blank_answers() -> None:
    assert [method.name for method in BUILTINS if method.marks_blank] == ["gradeflow/full-marks"]


@pytest.mark.parametrize(
    ("method", "settings", "question"),
    [
        ("gradeflow/key-points@1", {"points": ["x"]}, NumberQuestion()),
        ("gradeflow/number-range@1", {}, ChoiceQuestion(options=frozenset({"a"}))),
        ("gradeflow/multiple-choice@1", {"correct": ["a"]}, ValuesQuestion(value_types=["text"])),
    ],
)
def test_a_method_says_which_questions_it_cant_mark(
    method: str, settings: dict[str, object], question: object
) -> None:
    [(check, field, message)] = problems_of({"uses": method, "with": settings}, question)  # type: ignore[arg-type]
    assert (check, field) == ("", ())
    assert "can't mark a" in message
