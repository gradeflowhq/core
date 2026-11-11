import json
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gradeflow_core import InvalidError, Problem
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.reading import ListReading


def test_choice_reads_options_ignoring_capitals_unless_set_otherwise() -> None:
    question = ChoiceQuestion(options=("A", "b", "c"))
    assert question.options == ("a", "b", "c")
    assert question.read("A, B, C") == frozenset({"a", "b", "c"})


def test_choice_reads_with_a_custom_separator() -> None:
    question = ChoiceQuestion(options=("a", "b", "c"), reading=ListReading(separator="|"))
    options = question.read("Yes|NO|  Maybe ")
    assert options == frozenset({"yes", "no", "maybe"})


def test_choice_rejects_empty_option() -> None:
    with pytest.raises(ValidationError, match="Options can't be blank") as exc_info:
        ChoiceQuestion(options=("", "A"))
    assert exc_info.value.errors()[0]["loc"] == ("options",)


@pytest.mark.parametrize("options", [["A", "a"], ["a", " a "], ["a", "a"]])
def test_choice_rejects_options_read_alike(options: list[str]) -> None:
    with pytest.raises(InvalidError) as exc_info:
        ChoiceQuestion.load({"options": options})
    assert exc_info.value.problems == [
        Problem("Options must be different: “a” appears twice.", field="/options")
    ]


def test_choice_rejects_an_option_with_the_separator_in_it() -> None:
    with pytest.raises(InvalidError) as exc_info:
        ChoiceQuestion.load({"options": ["Yes, always", "No"]})
    assert exc_info.value.problems == [
        Problem("Options can't have the separator in them: “Yes, always” has “,”.", "", "/options")
    ]
    question = ChoiceQuestion(options=("Yes, always", "No"), reading=ListReading(separator=";"))
    assert question.read("Yes, always") == frozenset({"yes, always"})


def test_choice_options_are_read_like_answers() -> None:
    question = ChoiceQuestion(options=(" A", "b "))
    assert question.options == ("a", "b")


@pytest.mark.parametrize("kwargs", [{}, {"options": []}, {"options": set()}])
def test_choice_requires_at_least_one_option(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidError):
        ChoiceQuestion.load(kwargs)


def test_a_choice_answer_is_written_by_choosing_options() -> None:
    question = ChoiceQuestion(options=("a", "b", "c"))
    option_schema = question.answer_schema()["properties"]["option"]
    assert option_schema["enum"] == ["a", "b", "c"]
    assert question.answer_values("B") == {"option": "b"}
    assert question.answer_values("z") == {"option": None}
    assert question.answer_from({"option": "b"}) == "b"
    assert question.answer_from({"option": None}) == ""


def test_a_choice_of_several_options_takes_them_in_order() -> None:
    question = ChoiceQuestion(options=("a", "b", "c"), allow_multiple=True)
    option_schema = question.answer_schema()["properties"]["options"]
    assert (
        option_schema["type"],
        option_schema["uniqueItems"],
        option_schema["items"]["enum"],
    ) == (
        "array",
        True,
        ["a", "b", "c"],
    )
    assert question.answer_values("C, a, z") == {"options": ["a", "c"]}
    assert question.answer_from({"options": ["c", "a"]}) == "a,c"
    assert question.answer_from({"options": []}) == ""


@pytest.mark.parametrize("seed", ["1", "2", "3"])
def test_chosen_options_that_sort_alike_are_written_in_one_order_across_hash_seeds(
    seed: str,
) -> None:
    code = (
        "import json; from gradeflow_core import ChoiceQuestion; "
        "q = ChoiceQuestion(options=('1', '01', '001'), allow_multiple=True); "
        "print(json.dumps([q.normalize('1, 01, 001'), q.answer_values('1, 01, 001'), "
        "q.answer_schema()['properties']['options']['items']['enum']]))"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}, text=True
    )
    assert json.loads(output) == [
        "001, 01, 1",
        {"options": ["1", "01", "001"]},
        ["1", "01", "001"],
    ]


def test_choice_ignores_punctuation_when_set() -> None:
    question = ChoiceQuestion.load(
        {"options": ["(a)", "b."], "reading": {"ignore_punctuation": True}}
    )
    assert question.options == ("a", "b")
    assert question.read("(A), b") == frozenset({"a", "b"})


def test_options_keep_the_order_they_are_listed_in() -> None:
    question = ChoiceQuestion.load({"options": ["c", "A", "b"], "allow_multiple": True})
    assert question.options == ("c", "a", "b")
    assert question.dump()["options"] == ["c", "a", "b"]
    assert question.answer_schema()["properties"]["options"]["items"]["enum"] == ["c", "a", "b"]
    assert question.answer_values("B, C") == {"options": ["c", "b"]}
    assert question.answer_from({"options": ["b", "c"]}) == "c,b"


def test_options_must_be_listed() -> None:
    with pytest.raises(InvalidError) as exc_info:
        ChoiceQuestion.load({"options": {"a", "b"}})
    assert exc_info.value.problems == [Problem("Must be a list.", field="/options")]


def test_options_first_answered_come_after_the_listed_ones() -> None:
    question = ChoiceQuestion.load({"options": ["c", "a"]})
    assert question.accepting(["z, b", "a"]).options == ("c", "a", "b", "z")
