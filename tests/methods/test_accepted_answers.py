from typing import Any

import pytest

from gradeflow_core import BUILTINS, QuestionSet
from gradeflow_core.editing import Position
from gradeflow_core.ml import MISSING_MODEL, nli
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.reading import TextReading
from gradeflow_core.questions.text import TextQuestion

from .helpers import QID, describe, mark, mark_all, problems_of, proportions

METHODS = {method.ref: method for method in BUILTINS}

ACCEPTED = "gradeflow/accepted-answers@1"


def test_an_empty_list_accepts_nothing() -> None:
    check = {"uses": ACCEPTED, "with": {"answers": []}}
    assert describe(check) == "No accepted answers."
    assert proportions(mark_all(check, ["Canberra", ""])) == [0, 0]


def test_text_must_match_an_accepted_answer_as_the_question_reads_it() -> None:
    check = {"uses": ACCEPTED, "with": {"answers": ["Canberra"]}}
    assert proportions(mark_all(check, ["Canberra", "canberra", "Sydney"])) == [1, 1, 0]
    assert mark(check, "Sydney").feedback == "Expected “Canberra”."
    strict = TextQuestion(reading=TextReading(ignore_case=False))
    assert proportions(mark_all(check, ["Canberra", "canberra"], strict)) == [1, 0]


def test_similar_spelling_accepts_small_mistakes_and_says_how_close() -> None:
    check = {"uses": ACCEPTED, "with": {"answers": ["ileum"], "match": "spelling"}}
    exact, close, far = mark_all(check, ["ileum", "ilium", "colon"])
    assert (exact.proportion, exact.feedback) == (1, "Correct.")
    assert (close.proportion, close.feedback) == (1, "Accepted as “ileum”.")
    assert (far.proportion, far.feedback) == (0, "Expected “ileum”.")


@pytest.mark.parametrize(
    ("strictness", "proportions_given"),
    [("lenient", [1, 1, 1]), ("normal", [1, 1, 0]), ("strict", [1, 0, 0])],
)
def test_how_strict_sets_how_alike_the_spelling_must_be(
    strictness: str, proportions_given: list[int]
) -> None:
    check = {
        "uses": ACCEPTED,
        "with": {"answers": ["mitochondria"], "match": "spelling", "strictness": strictness},
    }
    answers = ["mitocondria", "mitokondria", "mitocondrai"]  # 92%, 83% and 75% alike
    assert proportions(mark_all(check, answers)) == proportions_given


@pytest.fixture
def judged(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str], str, float]]:
    """Calls to the meaning model, which says the second accepted answer matches "yes"."""
    calls: list[tuple[str, list[str], str, float]] = []

    def best_match(
        answer: str, references: list[str], context: str, cutoff: float
    ) -> tuple[int, bool]:
        calls.append((answer, list(references), context, cutoff))
        return 1, answer == "yes"

    monkeypatch.setattr(nli, "best_match", best_match)
    return calls


def test_same_meaning_reads_the_question_and_names_the_closest_answer(
    judged: list[tuple[str, list[str], str, float]],
) -> None:
    check = {
        "uses": ACCEPTED,
        "with": {"answers": ["first", "second"], "match": "meaning", "strictness": "strict"},
    }
    question = TextQuestion(description="  Why?\n")
    exact, close, far = mark_all(check, ["first", "yes", "no"], question)
    assert (exact.proportion, exact.feedback) == (1, "Correct.")
    assert (close.proportion, close.feedback) == (1, "Accepted as “second”.")
    assert (far.proportion, far.feedback) == (0, "Expected “first” or “second”.")
    assert judged == [
        ("yes", ["first", "second"], "Why?", 0.5),
        ("no", ["first", "second"], "Why?", 0.5),
    ]


def test_same_meaning_without_a_description_compares_the_answers_alone(
    judged: list[tuple[str, list[str], str, float]],
) -> None:
    check = {"uses": ACCEPTED, "with": {"answers": ["a", "b"], "match": "meaning"}}
    mark(check, "c")
    assert judged == [("c", ["a", "b"], "", 0.005)]


def test_same_meaning_without_the_ml_extra_is_a_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nli, "available", lambda: False)
    check = {"uses": ACCEPTED, "with": {"answers": ["a"], "match": "meaning"}}
    assert problems_of(check) == [("", ("match",), MISSING_MODEL)]
    assert problems_of({"uses": ACCEPTED, "with": {"answers": ["a"], "match": "spelling"}}) == []


def test_numbers_match_as_numbers_within_the_tolerance() -> None:
    number = NumberQuestion()
    exact = {"uses": ACCEPTED, "with": {"answers": ["42"]}}
    assert proportions(mark_all(exact, ["42", "42.0", "42 kg", "4.2e1", "43"], number)) == [
        1,
        1,
        1,
        1,
        0,
    ]
    assert mark(exact, "43", number).feedback == "Expected 42."
    about = {"uses": ACCEPTED, "with": {"answers": ["9.81"], "tolerance": 0.05}}
    assert proportions(mark_all(about, ["9.85", "9.9"], number)) == [1, 0]
    assert mark(about, "9.9", number).feedback == "Expected about 9.81."


def test_it_describes_itself() -> None:
    assert describe({"uses": ACCEPTED, "with": {"answers": ["a", "b"]}}) == "Matches `a` or `b`."
    spelling = {"uses": ACCEPTED, "with": {"answers": ["a"], "match": "spelling"}}
    assert describe(spelling) == "Matches `a` in similar spelling."
    same = {
        "uses": ACCEPTED,
        "with": {"answers": ["a"], "match": "meaning", "strictness": "strict"},
    }
    assert describe(same) == "Matches `a` in meaning (strict)."
    numbers = {"uses": ACCEPTED, "with": {"answers": ["1"], "tolerance": 0.5}}
    assert describe(numbers) == "Matches `1`, within `0.5`."


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
        "uses": ACCEPTED,
        "with": {
            "answers": ["ok", answer] if isinstance(question, TextQuestion) else ["1", answer]
        },
    }
    assert problems_of(check, question) == [("", ("answers", 1), message)]  # type: ignore[arg-type]


def test_its_form_fits_the_question() -> None:
    questions = QuestionSet({QID: TextQuestion(), "Q2": NumberQuestion()})

    def settings(question: str) -> dict[str, Any]:
        properties: dict[str, Any] = METHODS[ACCEPTED].settings_schema(
            Position(questions, question)
        )["properties"]
        return properties

    text, number = settings(QID), settings("Q2")
    assert set(text) == {"answers", "match", "strictness"}
    assert set(number) == {"answers", "tolerance"}
    assert text["answers"]["x-gradeflow"] == {"accepts_answers": True}
    assert text["strictness"]["x-gradeflow"]["applies_when"] == {"match": ["spelling", "meaning"]}
