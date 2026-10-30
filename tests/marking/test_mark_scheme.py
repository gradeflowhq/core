from typing import Any

import pytest

from gradeflow_core import (
    BUILTINS,
    Check,
    GradeFlowError,
    InvalidError,
    MarkScheme,
    Method,
    Problem,
    QuestionSet,
)
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion
from tests.methods.helpers import key_point

QUESTIONS = QuestionSet(
    {
        "Q1": TextQuestion(),
        "Q2": NumberQuestion(),
        "Q3": ValuesQuestion(value_types=["text", "number"]),
        "Q4": ChoiceQuestion(options=frozenset({"a", "b"})),
    }
)


def problems(*markings: dict[str, Any]) -> list[tuple[str, tuple[str | int, ...], str]]:
    scheme = MarkScheme.load(list(markings))
    return [
        (problem.check, problem.field, problem.message) for problem in scheme.problems(QUESTIONS)
    ]


def test_a_mark_scheme_that_can_mark_has_no_problems() -> None:
    assert (
        problems(
            key_point("a", question="Q1"),
            {"question": "Q4", "uses": "gradeflow/multiple-choice@1", "with": {"correct": ["a"]}},
        )
        == []
    )


def test_a_question_is_marked_by_one_marking_the_first() -> None:
    assert problems(key_point("a", question="Q1"), key_point("a", question="Q1")) == [
        ("markings.1", (), "Q1 is marked twice.")
    ]


def test_checks_mark_only_where_they_can() -> None:
    assert problems(
        key_point("a", question="Q9"),
        {"question": "Q2", "uses": "gradeflow/key-points@1", "with": {"points": ["a"]}},
        key_point("a"),
        {"question": "Q3", "uses": "gradeflow/multiple-choice@1", "with": {"correct": ["a"]}},
    ) == [
        ("markings.0", ("question",), "Q9 isn't in the supplied questions."),
        ("markings.1", (), "Key Points can't mark a number question."),
        ("markings.2", ("question",), "Choose the question it checks."),
        ("markings.3", (), "Multiple Choice can't mark a multiple values question."),
    ]


def test_problems_in_a_question_group_are_located_in_its_checks() -> None:
    group = {
        "uses": "gradeflow/combination@1",
        "with": {
            "checks": [
                key_point("a"),
                key_point("a", question="Q1", value=1),
                key_point("a", question="Q3", value=3),
                key_point("a", question="Q3", value=1, id="x"),
                key_point("a", question="Q3", value=1, id="x"),
            ]
        },
    }
    assert problems(group) == [
        ("markings.0.with.checks.0", ("question",), "Choose the question it checks."),
        (
            "markings.0.with.checks.1",
            ("value",),
            "Only a question with several values has values to check.",
        ),
        ("markings.0.with.checks.2", ("value",), "The answer has 2 values."),
        ("markings.0.with.checks.4", ("id",), "Another check here is called x."),
    ]


def test_a_value_is_marked_as_its_type() -> None:
    number_value = {
        "question": "Q3",
        "value": 2,
        "uses": "gradeflow/key-points@1",
        "with": {"points": ["a"]},
    }
    assert problems({"uses": "gradeflow/combination@1", "with": {"checks": [number_value]}}) == [
        ("markings.0.with.checks.0", (), "Key Points can't mark a number value.")
    ]


def test_markings_read_their_marked_questions_and_dependencies() -> None:
    conditional = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [key_point("a", question="Q1")],
            "then": [key_point("a", question="Q2")],
        },
    }
    scheme = MarkScheme.load([conditional, key_point("a", question="Q4")])
    assert [
        (marking.questions_read(), marking.questions_marked()) for marking in scheme.markings
    ] == [({"Q1", "Q2"}, {"Q2"}), ({"Q4"}, {"Q4"})]


@pytest.mark.parametrize(
    "method", [method for method in BUILTINS if method.examples], ids=lambda method: method.name
)
def test_checks_written_without_their_defaults_read_back_the_same(method: Method[Any]) -> None:
    for example in method.examples:
        data: dict[str, Any] = {"uses": method.ref, "with": example.with_}
        if len(example.questions) == 1:
            data["question"] = next(iter(example.questions))
        written = Check.load(data).dump()
        assert Check.load(written).dump() == written


def test_loading_markings_that_arent_checks_says_where_each_problem_is() -> None:
    markings = [
        {"question": "Q1", "uses": "gradeflow/full-marks@1"},
        {"question": "Q2", "uses": "gradeflow/nope@1"},
        {"question": "Q3", "uses": "gradeflow/combination@1", "with": {"checks": [{}]}},
    ]
    with pytest.raises(InvalidError) as caught:
        MarkScheme.load(markings)
    assert caught.value.problems == [
        Problem("There's no method gradeflow/nope@1.", "markings.1", ("uses",)),
        Problem("Choose a method.", "markings.2.with.checks.0", ("uses",)),
    ]
    assert isinstance(caught.value, GradeFlowError)


def test_a_mark_scheme_dumps_as_it_loads() -> None:
    markings = [
        {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {"answers": ["a"]}},
        {"question": "Q2", "uses": "gradeflow/full-marks@1"},
    ]
    scheme = MarkScheme.load(markings)
    assert scheme.dump() == markings
    assert MarkScheme.load(scheme.dump()) == scheme


def test_a_mark_scheme_is_written_and_read_as_yaml() -> None:
    scheme = MarkScheme.load([key_point("a", question="Q1")])
    assert MarkScheme.from_yaml(scheme.to_yaml()) == scheme
    with pytest.raises(InvalidError, match="Line 1: "):
        MarkScheme.from_yaml("- question: [Q1")
    with pytest.raises(InvalidError, match="markings: Must be a list"):
        MarkScheme.load("- question: Q1")
