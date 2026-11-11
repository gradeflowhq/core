from typing import Any

import pytest

from gradeflow_core import BUILTINS, QuestionSet
from gradeflow_core.editing import Position
from gradeflow_core.models import nli
from gradeflow_core.models.local import MISSING_MODEL_MESSAGE
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.reading import TextReading
from gradeflow_core.questions.text import TextQuestion

from .helpers import QUESTION_ID, describe, mark, mark_all, problems_of, proportions

ACCEPTED_ANSWERS = "gradeflow/accepted-answers@1"


def test_an_empty_list_accepts_nothing() -> None:
    check = {"uses": ACCEPTED_ANSWERS, "with": {"answers": []}}
    assert describe(check) == "No accepted answers."
    assert proportions(mark_all(check, ["Canberra", ""])) == [0, 0]


def test_text_must_match_an_accepted_answer_as_the_question_reads_it() -> None:
    check = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["Canberra"]}}
    assert proportions(mark_all(check, ["Canberra", "canberra", "Sydney"])) == [1, 1, 0]
    assert mark(check, "Sydney").feedback == "Expected “Canberra”."
    case_sensitive = TextQuestion(reading=TextReading(ignore_case=False))
    assert proportions(mark_all(check, ["Canberra", "canberra"], case_sensitive)) == [1, 0]


def test_similar_spelling_accepts_misspellings_and_says_how_close() -> None:
    check = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["ileum"], "match": "spelling"}}
    exact_result, close_result, wrong_result = mark_all(check, ["ileum", "ilium", "colon"])
    assert (exact_result.proportion, exact_result.feedback) == (1, "Correct.")
    assert (close_result.proportion, close_result.feedback) == (1, "Accepted as “ileum”.")
    assert (wrong_result.proportion, wrong_result.feedback) == (0, "Expected “ileum”.")


@pytest.mark.parametrize(
    ("strictness", "expected"),
    [("lenient", [1, 1, 1]), ("normal", [1, 1, 0]), ("strict", [1, 0, 0])],
)
def test_how_strict_sets_how_alike_the_spelling_must_be(
    strictness: str, expected: list[int]
) -> None:
    check = {
        "uses": ACCEPTED_ANSWERS,
        "with": {"answers": ["mitochondria"], "match": "spelling", "strictness": strictness},
    }
    answers = ["mitocondria", "mitokondria", "mitocondrai"]  # 92%, 83% and 75% alike
    assert proportions(mark_all(check, answers)) == expected


@pytest.fixture
def meaning_model_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str], str, float]]:
    """Calls to the meaning model, which says the second accepted answer matches "yes"."""
    calls: list[tuple[str, list[str], str, float]] = []

    def first_match_index(
        answer: str, references: list[str], context: str, cutoff: float
    ) -> int | None:
        calls.append((answer, list(references), context, cutoff))
        return 1 if answer == "yes" else None

    monkeypatch.setattr(nli, "first_match_index", first_match_index)
    return calls


def test_same_meaning_reads_the_question_and_names_the_closest_answer(
    meaning_model_calls: list[tuple[str, list[str], str, float]],
) -> None:
    check = {
        "uses": ACCEPTED_ANSWERS,
        "with": {"answers": ["first", "second"], "match": "meaning", "strictness": "strict"},
    }
    question = TextQuestion(description="  Why?\n")
    exact_result, close_result, wrong_result = mark_all(check, ["first", "yes", "no"], question)
    assert (exact_result.proportion, exact_result.feedback) == (1, "Correct.")
    assert (close_result.proportion, close_result.feedback) == (1, "Accepted as “second”.")
    assert (wrong_result.proportion, wrong_result.feedback) == (0, "Expected “first” or “second”.")
    assert meaning_model_calls == [
        ("yes", ["first", "second"], "Why?", 0.5),
        ("no", ["first", "second"], "Why?", 0.5),
    ]


def test_same_meaning_without_a_description_compares_the_answers_alone(
    meaning_model_calls: list[tuple[str, list[str], str, float]],
) -> None:
    check = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["a", "b"], "match": "meaning"}}
    mark(check, "c")
    assert meaning_model_calls == [("c", ["a", "b"], "", 0.005)]


def test_same_meaning_without_the_meaning_model_installed_is_a_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nli, "is_available", lambda: False)
    check = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["a"], "match": "meaning"}}
    assert problems_of(check) == [("", "/with/match", MISSING_MODEL_MESSAGE)]
    assert (
        problems_of({"uses": ACCEPTED_ANSWERS, "with": {"answers": ["a"], "match": "spelling"}})
        == []
    )


def test_numbers_match_as_numbers_within_the_tolerance() -> None:
    number = NumberQuestion()
    number_check = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["42"]}}
    assert proportions(mark_all(number_check, ["42", "42.0", "42 kg", "4.2e1", "43"], number)) == [
        1,
        1,
        1,
        1,
        0,
    ]
    assert mark(number_check, "43", number).feedback == "Expected 42."
    with_tolerance = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["9.81"], "tolerance": 0.05}}
    assert proportions(mark_all(with_tolerance, ["9.85", "9.9"], number)) == [1, 0]
    assert mark(with_tolerance, "9.9", number).feedback == "Expected about 9.81."


def test_numbers_are_equal_only_up_to_rounding_relative_to_their_size() -> None:
    number = NumberQuestion()
    tiny = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["6.63e-34"]}}
    assert proportions(mark_all(tiny, ["6.63e-34", "9e-34", "0"], number)) == [1, 0, 0]
    tenth = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["0.3"], "tolerance": 0.1}}
    assert proportions(mark_all(tenth, ["0.4", "0.2", "0.41"], number)) == [1, 1, 0]
    with_tolerance = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["9.81"], "tolerance": 0.05}}
    assert proportions(mark_all(with_tolerance, ["9.86", "9.76"], number)) == [1, 1]


def test_it_describes_itself() -> None:
    assert (
        describe({"uses": ACCEPTED_ANSWERS, "with": {"answers": ["a", "b"]}})
        == "Matches `a` or `b`."
    )
    spelling = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["a"], "match": "spelling"}}
    assert describe(spelling) == "Matches `a` in similar spelling."
    same_meaning = {
        "uses": ACCEPTED_ANSWERS,
        "with": {"answers": ["a"], "match": "meaning", "strictness": "strict"},
    }
    assert describe(same_meaning) == "Matches `a` in meaning (strict)."
    numbers = {"uses": ACCEPTED_ANSWERS, "with": {"answers": ["1"], "tolerance": 0.5}}
    assert describe(numbers) == "Matches `1`, within `0.5`."


def test_long_lists_of_answers_are_listed_one_per_line() -> None:
    answers = [
        "Heat is released to the surroundings",
        "Energy is given out, so the surroundings warm up",
    ]
    same_meaning = {"uses": ACCEPTED_ANSWERS, "with": {"answers": answers, "match": "meaning"}}
    assert describe(same_meaning) == (
        "Matches any of these in meaning:\n"
        "- `Heat is released to the surroundings`\n"
        "- `Energy is given out, so the surroundings warm up`"
    )
    numbers = {
        "uses": ACCEPTED_ANSWERS,
        "with": {"answers": [str(digit) * 9 for digit in range(1, 10)], "tolerance": 0.5},
    }
    assert describe(numbers).startswith("Matches any of these, within `0.5`:\n- `111111111`\n")


@pytest.mark.parametrize(
    ("question", "answer", "message"),
    [
        (TextQuestion(), "N/A", "It reads as a blank answer."),
        (NumberQuestion(), "about 9", "Couldn't read “about 9” as a number."),
    ],
)
def test_an_accepted_answer_the_question_cant_read_is_a_problem(
    question: object, answer: str, message: str
) -> None:
    check = {
        "uses": ACCEPTED_ANSWERS,
        "with": {
            "answers": ["ok", answer] if isinstance(question, TextQuestion) else ["1", answer]
        },
    }
    assert problems_of(check, question) == [("", "/with/answers/1", message)]  # type: ignore[arg-type]


def test_its_settings_fit_the_question() -> None:
    questions = QuestionSet({QUESTION_ID: TextQuestion(), "Q2": NumberQuestion()})

    def settings_properties(question_id: str) -> dict[str, Any]:
        properties: dict[str, Any] = BUILTINS.find(ACCEPTED_ANSWERS).settings_schema(
            Position(questions, question_id)
        )["properties"]
        return properties

    text_settings = settings_properties(QUESTION_ID)
    number_settings = settings_properties("Q2")
    assert set(text_settings) == {"answers", "match", "strictness"}
    assert set(number_settings) == {"answers", "tolerance"}
    assert text_settings["answers"]["x-gradeflow"] == {"lists_answers": True}
    assert text_settings["strictness"]["x-gradeflow"]["applies_when"] == {
        "match": ["spelling", "meaning"]
    }
