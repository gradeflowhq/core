from typing import Any

import pytest

from gradeflow_core.ml import MISSING_MODEL_MESSAGE, nli
from gradeflow_core.questions.text import TextQuestion

from .helpers import describe, mark, mark_all, problems_of, proportions

KEY_POINT = "gradeflow/key-point@1"
KEY_POINTS = "gradeflow/key-points@1"

type MeaningModelCall = tuple[str, str, str, float]


def key_points(*points: str, **settings: Any) -> dict[str, Any]:
    return {"uses": KEY_POINTS, "with": {"points": list(points), **settings}}


@pytest.fixture
def meaning_model_calls(monkeypatch: pytest.MonkeyPatch) -> list[MeaningModelCall]:
    """Calls to the meaning model, which says an answer makes a point when it contains it."""
    calls: list[MeaningModelCall] = []

    def entails(answer: str, point: str, context: str, cutoff: float) -> bool:
        calls.append((answer, point, context, cutoff))
        return point.lower() in answer.lower()

    monkeypatch.setattr(nli, "entails", entails)
    return calls


def test_a_point_is_found_anywhere_ignoring_capitals_or_as_whole_words() -> None:
    check = {"uses": KEY_POINT, "with": {"point": "for"}}
    found, within_a_word, missing = mark_all(check, ["for loop", "Format", "nothing"])
    assert (found.proportion, found.feedback) == (1, "Has “for”.")
    assert (within_a_word.proportion, missing.proportion, missing.feedback) == (
        1,
        0,
        "Missing “for”.",
    )
    assert describe(check) == "Includes `for`."
    whole_words = {"uses": KEY_POINT, "with": {"point": "for", "whole_words": True}}
    assert proportions(mark_all(whole_words, ["for loop", "format"])) == [1, 0]
    assert describe(whole_words) == "Includes the whole word `for`."


@pytest.mark.parametrize("match", ["exact", "spelling"])
def test_a_point_is_read_as_the_question_reads_answers(match: str) -> None:
    question = TextQuestion.load({"reading": {"ignore_punctuation": True}})
    check = {"uses": KEY_POINT, "with": {"point": "CO2.", "match": match}}
    found = mark(check, "Plants take in co2", question)
    assert (found.proportion, found.feedback) == (1, "Has “CO2”.")


def test_similar_spelling_compares_runs_of_as_many_words() -> None:
    check = {"uses": KEY_POINT, "with": {"point": "carbon dioxide", "match": "spelling"}}
    exact, close, far, short = mark_all(
        check, ["carbon dioxide.", "They take in carbon dioxyde!", "carbon and dioxide", "carbon"]
    )
    assert (exact.proportion, exact.feedback) == (1, "Has “carbon dioxide”.")
    assert (close.proportion, close.feedback) == (1, "Has “carbon dioxide” (as “carbon dioxyde”).")
    assert (far.proportion, far.feedback) == (0, "Missing “carbon dioxide”.")
    assert short.proportion == 0
    assert describe(check) == "Includes `carbon dioxide` in similar spelling."


def test_similar_spelling_ignores_punctuation_around_the_points_words_too() -> None:
    check = {"uses": KEY_POINT, "with": {"point": "H2O.", "match": "spelling"}}
    result = mark(check, "Water is H2O.")
    assert (result.proportion, result.feedback) == (1, "Has “H2O”.")


@pytest.mark.parametrize(
    ("strictness", "expected"),
    [("lenient", [1, 1, 1]), ("normal", [1, 1, 0]), ("strict", [1, 0, 0])],
)
def test_how_strict_sets_how_alike_the_words_must_be(strictness: str, expected: list[int]) -> None:
    check = {
        "uses": KEY_POINT,
        "with": {"point": "mitochondria", "match": "spelling", "strictness": strictness},
    }
    answers = ["the mitocondria", "the mitokondria", "the mitocondrai"]  # 92%, 83% and 75% alike
    assert proportions(mark_all(check, answers)) == expected


def test_same_meaning_reads_the_question_and_quotes_the_point(
    meaning_model_calls: list[MeaningModelCall],
) -> None:
    check = {"uses": KEY_POINT, "with": {"point": "Ice is less dense", "match": "meaning"}}
    has_point, lacks_point = mark_all(
        check,
        ["ice is less dense, so it floats", "Ice is solid"],
        TextQuestion(description=" Why?\n"),
    )
    assert (has_point.proportion, has_point.feedback) == (1, "Has “Ice is less dense”.")
    assert (lacks_point.proportion, lacks_point.feedback) == (0, "Missing “Ice is less dense”.")
    assert meaning_model_calls == [
        ("ice is less dense, so it floats", "Ice is less dense", "Why?", 0.1),
        ("ice is solid", "Ice is less dense", "Why?", 0.1),
    ]
    assert describe(check) == "Includes `Ice is less dense` in meaning."


@pytest.mark.parametrize(
    ("strictness", "cutoff"), [("lenient", -0.05), ("normal", 0.1), ("strict", 0.9)]
)
def test_how_strict_sets_the_meaning_cutoff(
    meaning_model_calls: list[MeaningModelCall], strictness: str, cutoff: float
) -> None:
    mark(
        {"uses": KEY_POINT, "with": {"point": "a", "match": "meaning", "strictness": strictness}},
        "b",
    )
    assert meaning_model_calls == [("b", "a", "", cutoff)]


def test_without_the_meaning_model_installed_same_meaning_is_a_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nli, "available", lambda: False)
    assert problems_of({"uses": KEY_POINT, "with": {"point": "a", "match": "meaning"}}) == [
        ("", ("match",), MISSING_MODEL_MESSAGE)
    ]
    problems = problems_of(key_points("a", "b", match="meaning"))
    assert [message for _, _, message in problems] == [
        MISSING_MODEL_MESSAGE,
        MISSING_MODEL_MESSAGE,
    ]
    assert problems_of({"uses": KEY_POINT, "with": {"point": "a"}}) == []


def test_key_points_count_each_point_or_add_up_a_mark_for_each() -> None:
    check = key_points("light", "water", "glucose")
    assert proportions(
        mark_all(check, ["light, water and glucose", "light and sugar", "soil"])
    ) == [
        1,
        0.3333,
        0,
    ]
    assert mark(check, "light").feedback == "Has “light”.\nMissing “water”.\nMissing “glucose”."
    assert describe(check) == "Includes `light`, `water` and `glucose`, a share for each."
    assert describe(key_points("light")) == "Includes `light`."
    a_mark_each = key_points("light", "water", "glucose", combine="total")
    full, half = mark_all(a_mark_each, ["light and water", "light"], TextQuestion(max_marks=2))
    assert (full.proportion, full.feedback) == (1, "Has “light”.\nHas “water”.")
    assert half.proportion == 0.5
    assert describe(a_mark_each) == "Includes `light`, `water` and `glucose`, a mark each."


def test_key_points_any_one_of_which_is_missing_names_them_all() -> None:
    check = key_points("light", "sunlight", combine="any")
    assert mark(check, "water").feedback == "Missing “light” or “sunlight”."
    assert mark(check, "sunlight").feedback == "Has “light”."
    assert describe(check) == "Includes `light` or `sunlight`."


def test_key_points_none_of_which_may_appear_say_which_did() -> None:
    check = key_points("for", "while", combine="none")
    assert [mark(check, answer).feedback for answer in ("n % 10", "for x in y")] == [
        "Has none of “for” or “while”.",
        "Not allowed. Has “for”.",
    ]
    assert describe(check) == "Doesn't include `for` and `while`."
    whole_words = key_points("for", "while", combine="none", whole_words=True)
    assert mark(whole_words, "format").proportion == 1
    assert describe(whole_words) == "Doesn't include the whole words `for` and `while`."


def test_key_points_pass_how_they_match_to_each_point(
    meaning_model_calls: list[MeaningModelCall],
) -> None:
    spelling = key_points("photosynthesis", "chlorophyll", match="spelling")
    assert mark(spelling, "Fotosynthesis needs chlorofyll").feedback == (
        "Has “photosynthesis” (as “fotosynthesis”).\nHas “chlorophyll” (as “chlorofyll”)."
    )
    assert describe(spelling) == (
        "Includes `photosynthesis` and `chlorophyll` in similar spelling, a share for each."
    )
    same_meaning = key_points(
        "ice is less dense", "water expands", match="meaning", strictness="strict"
    )
    half = mark(same_meaning, "Water expands")
    assert (half.proportion, half.feedback) == (
        0.5,
        "Missing “ice is less dense”.\nHas “water expands”.",
    )
    assert {cutoff for *_, cutoff in meaning_model_calls} == {0.9}
    assert describe(same_meaning) == (
        "Includes `ice is less dense` and `water expands` in meaning (strict), a share for each."
    )


@pytest.mark.parametrize(
    ("settings", "heading"),
    [
        ({"combine": "all"}, "Includes all of these:"),
        ({"combine": "any"}, "Includes any of these:"),
        ({"combine": "none", "whole_words": True}, "Doesn't include all of these as whole words:"),
        (
            {"combine": "proportional", "match": "meaning"},
            "Includes these in meaning, a share for each:",
        ),
        (
            {"combine": "total", "match": "spelling", "strictness": "lenient"},
            ("Includes these in similar spelling (lenient), a mark each:"),
        ),
    ],
)
def test_long_points_are_listed_one_per_line(settings: dict[str, Any], heading: str) -> None:
    points = [
        "Ice is less dense than liquid water",
        "The hydrogen-bonded lattice holds molecules apart",
    ]
    assert describe(key_points(*points, **settings)) == (
        f"{heading}\n- `{points[0]}`\n- `{points[1]}`"
    )
