from copy import deepcopy
from typing import Any

import pytest

from gradeflow_core import BUILTINS, Methods
from gradeflow_core.editing import Decision, EditingError
from gradeflow_core.methods.answers.accepted_answers import AcceptedAnswers
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from ..methods.helpers import accepted_answer
from .helpers import (
    ACCEPTS_9_81,
    answer_proportions,
    check_of,
    decisions_of,
    marked_by_example,
)

ACCEPTS_X = accepted_answer("x")
UNKNOWN_MARKS_BY: tuple[object, ...] = (None, "nonsense", [])


@pytest.mark.parametrize("action", ["full_marks", "partial_marks", "not_allowed"])
def test_nested_decisions_are_read_and_cancelled(action: Decision) -> None:
    marking = marked_by_example(ACCEPTS_9_81, ["10"], action)
    marking = marked_by_example(marking, ["9.8"], "partial_marks")
    before = deepcopy(marking)
    assert decisions_of(marking) == {"9.81": "full_marks", "10": action, "9.8": "partial_marks"}
    cancelled = marked_by_example(marking, ["10"], "cancel")
    assert decisions_of(cancelled) == {"9.81": "full_marks", "9.8": "partial_marks"}
    assert answer_proportions(cancelled) == [1, 0, 0.5, 0]
    assert marking == before


@pytest.mark.parametrize("action", ["full_marks", "partial_marks", "not_allowed"])
def test_cancelling_an_override_restores_the_remaining_method(action: Decision) -> None:
    marking = {"question": "Q1", "uses": "gradeflow/number-range@1", "with": {"min_value": 0}}
    with_decision = marked_by_example(marking, ["10"], action)
    assert decisions_of(with_decision) == {"10": action}
    cancelled = marked_by_example(with_decision, ["10"], "cancel")
    assert decisions_of(cancelled) == {}
    assert answer_proportions(cancelled) == [1, 0, 1, 1]
    assert cancelled == marking


def test_full_marks_method_keeps_an_explicit_full_decision() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/full-marks@1"}
    with_full_decision = marked_by_example(marking, ["10"], "full_marks")
    assert decisions_of(with_full_decision) == {"10": "full_marks"}
    assert marked_by_example(with_full_decision, ["10"], "cancel") == marking


def test_equivalent_answers_share_and_cancel_a_decision() -> None:
    question = NumberQuestion()
    marking = marked_by_example(ACCEPTS_9_81, ["10.0"], "not_allowed", question=question)
    assert decisions_of(marking, question=question) == {
        "9.81": "full_marks",
        "10": "not_allowed",
    }
    cancelled = marked_by_example(marking, ["1e1"], "cancel", question=question)
    assert decisions_of(cancelled, question=question) == {"9.81": "full_marks"}


def test_only_listed_answers_count_even_with_tolerance() -> None:
    marking = {**ACCEPTS_9_81, "with": {"answers": ["9.81", "10"], "tolerance": 0.02}}
    assert decisions_of(marking) == {"9.81": "full_marks", "10": "full_marks"}
    cancelled = marked_by_example(marking, ["9.81"], "cancel")
    assert cancelled == {**ACCEPTS_9_81, "with": {"answers": ["10"], "tolerance": 0.02}}
    assert answer_proportions(cancelled) == [0, 0, 0, 1]


@pytest.mark.parametrize(
    ("method", "settings"),
    [
        ("gradeflow/key-points", {"points": ["10"]}),
        ("gradeflow/pattern", {"pattern": "10"}),
        ("gradeflow/accepted-answers", {"answers": ["10"], "match": "spelling"}),
        ("gradeflow/accepted-answers", {"answers": ["10"], "match": "meaning"}),
        ("sample/custom@1", {"data": ACCEPTS_9_81}),
    ],
)
def test_general_methods_do_not_make_explicit_decisions(
    method: str, settings: dict[str, Any]
) -> None:
    assert decisions_of({"uses": method, "with": settings}) == {}


def test_cancellation_does_not_touch_other_questions_or_weight_zero_checks() -> None:
    children = [
        {**ACCEPTS_9_81, "question": "Q2"},
        {**ACCEPTS_9_81, "weight": 0},
        {**ACCEPTS_9_81, "input": "{{ values[0] }}"},
    ]
    marking = {"question": "Q1", "uses": "gradeflow/workflow@1", "with": {"steps": children}}
    assert decisions_of(marking) == {}
    assert marked_by_example(marking, ["9.81"], "cancel") == marking


def test_recursive_cancellation_keeps_conditions_and_weights() -> None:
    prerequisite = {**ACCEPTS_9_81, "id": "prerequisite", "weight": 0}
    child = {**ACCEPTS_9_81, "if": "{{ steps.prerequisite.output.correct }}", "weight": 0.25}
    marking = {
        "question": "Q1",
        "uses": "gradeflow/workflow@1",
        "with": {"marks_by": "any_correct", "steps": [prerequisite, child]},
    }
    assert decisions_of(marking) == {"9.81": "partial_marks"}
    cancelled = marked_by_example(marking, ["9.81"], "cancel")
    assert decisions_of(cancelled) == {}
    remaining_step = cancelled["with"]["steps"][1]
    assert remaining_step["if"] == "{{ steps.prerequisite.output.correct }}"
    assert remaining_step["weight"] == 0.25


def test_double_negation_and_added_up_decisions() -> None:
    check = accepted_answer("10")
    for _ in range(2):
        check = {"uses": "gradeflow/not-allowed@1", "with": {"check": check}}
    assert decisions_of(check) == {"10": "full_marks"}
    accepts_eleven = accepted_answer("11")
    proportional: dict[str, Any] = {
        "uses": "gradeflow/workflow@1",
        "with": {"marks_by": "proportional", "steps": [check, accepts_eleven]},
    }
    assert decisions_of(proportional) == {"10": "partial_marks", "11": "partial_marks"}
    cancelled = marked_by_example(proportional, ["10"], "cancel")
    assert decisions_of(cancelled) == {"11": "partial_marks"}
    total_workflow = {
        "uses": "gradeflow/workflow@1",
        "with": {"marks_by": "total", "steps": [check]},
    }
    assert decisions_of(total_workflow, question=NumberQuestion(max_marks=1)) == {
        "10": "full_marks"
    }
    assert decisions_of(total_workflow, question=NumberQuestion(max_marks=2)) == {
        "10": "partial_marks"
    }


@pytest.mark.parametrize(("weights", "max_marks"), [((0.01, 0.29, 0.7), 1), ((0.3, 1.4, 2.3), 4)])
def test_marks_adding_up_to_full_marks_are_full_despite_rounding(
    weights: tuple[float, ...], max_marks: float
) -> None:
    checks = [{**ACCEPTS_X, "weight": weight} for weight in weights]
    total_workflow = {
        "uses": "gradeflow/workflow@1",
        "with": {"marks_by": "total", "steps": checks},
    }
    question = TextQuestion(max_marks=max_marks)
    assert decisions_of(total_workflow, question=question) == {"x": "full_marks"}


def test_cancel_preserves_other_proportions_of_the_marks() -> None:
    marking = {
        "question": "Q1",
        "uses": "gradeflow/workflow@1",
        "with": {
            "marks_by": "proportional",
            "steps": [
                accepted_answer("10"),
                accepted_answer("9.81", weight=3),
            ],
        },
    }
    assert answer_proportions(marking) == [0.75, 0, 0, 0.25]
    cancelled = marked_by_example(marking, ["10"], "cancel")
    assert answer_proportions(cancelled) == [0.75, 0, 0, 0]
    assert decisions_of(cancelled) == {"9.81": "partial_marks"}


@pytest.mark.parametrize(
    "marking",
    [
        {},
        {"uses": "gradeflow/not-allowed@1", "with": {}},
        {"uses": "gradeflow/accepted-answers@1", "with": {}},
    ],
)
def test_an_incomplete_draft_has_no_decisions(marking: dict[str, Any]) -> None:
    assert decisions_of(marking) == {}


def test_cancelling_an_absent_decision_changes_nothing() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/workflow@1", "with": {"steps": [ACCEPTS_9_81]}}
    assert marked_by_example(marking, ["10"], "cancel") == marking
    cancelled = marked_by_example(marking, ["9.81"], "cancel")
    assert marked_by_example(cancelled, ["9.81"], "cancel") == cancelled


class AcceptedAnswersTwo(AcceptedAnswers):
    version = 2


def test_new_checks_use_the_latest_version_and_decisions_read_every_version() -> None:
    methods = Methods([*BUILTINS, AcceptedAnswersTwo()])
    marking = marked_by_example(ACCEPTS_9_81, ["10"], "full_marks", methods=methods)
    assert marking == {
        "question": "Q1",
        "uses": "gradeflow/workflow@1",
        "with": {
            "marks_by": "any_correct",
            "steps": [
                {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["9.81"]}},
                {"uses": "gradeflow/accepted-answers@2", "with": {"answers": ["10"]}},
            ],
        },
    }
    assert decisions_of(marking, methods=methods) == {"9.81": "full_marks", "10": "full_marks"}
    assert marked_by_example(marking, ["10"], "cancel", methods=methods) == ACCEPTS_9_81


def test_deciding_needs_the_methods_decisions_are_written_with() -> None:
    with pytest.raises(EditingError, match="There's no method gradeflow/accepted-answers"):
        decisions_of(ACCEPTS_9_81, methods=Methods())


@pytest.mark.parametrize(
    "marking",
    [
        {"question": "Q1", "uses": "gradeflow/not-allowed@1", "with": {"check": "nonsense"}},
        *(
            {
                "question": "Q1",
                "uses": "gradeflow/workflow@1",
                "with": {"marks_by": marks_by, "steps": [check_of(ACCEPTS_9_81)]},
            }
            for marks_by in UNKNOWN_MARKS_BY
        ),
        {
            "question": "Q1",
            "uses": "gradeflow/workflow@1",
            "with": {"steps": ["nonsense", {**check_of(ACCEPTS_9_81), "weight": "heavy"}]},
        },
    ],
)
def test_a_check_written_so_it_cant_be_read_decides_nothing(marking: dict[str, Any]) -> None:
    assert decisions_of(marking) == {}


def test_the_readable_checks_of_a_marking_still_decide() -> None:
    marking = {
        "question": "Q1",
        "uses": "gradeflow/workflow@1",
        "with": {
            "marks_by": "any_correct",
            "steps": [
                {"uses": "gradeflow/not-allowed@1", "with": {"check": 5}},
                check_of(ACCEPTS_9_81),
            ],
        },
    }
    assert decisions_of(marking) == {"9.81": "full_marks"}
