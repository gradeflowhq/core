from copy import deepcopy
from typing import Any

import pytest

from gradeflow_core import BUILTINS, Methods
from gradeflow_core.editing import Decision, EditingError, answer_decisions, mark_by_example
from gradeflow_core.methods.answers.accepted_answers import AcceptedAnswers
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from .test_by_example import ACCEPTS_9_81, answer_proportions

ACCEPTS_X = {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["x"]}}


@pytest.mark.parametrize("action", ["full_marks", "part_marks", "not_allowed"])
def test_nested_decisions_are_read_and_cancelled(action: Decision) -> None:
    marking = mark_by_example(ACCEPTS_9_81, ["10"], action)
    marking = mark_by_example(marking, ["9.8"], "part_marks")
    before = deepcopy(marking)
    assert answer_decisions(marking) == {"9.81": "full_marks", "10": action, "9.8": "part_marks"}
    cancelled = mark_by_example(marking, ["10"], "cancel")
    assert answer_decisions(cancelled) == {"9.81": "full_marks", "9.8": "part_marks"}
    assert answer_proportions(cancelled) == [1, 0, 0.5, 0]
    assert marking == before


def test_cancelling_an_override_restores_the_remaining_method() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/number-range@1", "with": {"min_value": 0}}
    for action in ("full_marks", "part_marks", "not_allowed"):
        with_decision = mark_by_example(marking, ["10"], action)
        assert answer_decisions(with_decision) == {"10": action}
        cancelled = mark_by_example(with_decision, ["10"], "cancel")
        assert answer_decisions(cancelled) == {}
        assert answer_proportions(cancelled) == [1, 0, 1, 1]
        assert cancelled == marking


def test_full_marks_method_keeps_an_explicit_full_decision() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/full-marks@1"}
    with_full_decision = mark_by_example(marking, ["10"], "full_marks")
    assert answer_decisions(with_full_decision) == {"10": "full_marks"}
    assert mark_by_example(with_full_decision, ["10"], "cancel") == marking


def test_equivalent_answers_share_and_cancel_a_decision() -> None:
    question = NumberQuestion()
    marking = mark_by_example(ACCEPTS_9_81, ["10.0"], "not_allowed", question=question)
    assert answer_decisions(marking, question=question) == {
        "9.81": "full_marks",
        "10": "not_allowed",
    }
    cancelled = mark_by_example(marking, ["1e1"], "cancel", question=question)
    assert answer_decisions(cancelled, question=question) == {"9.81": "full_marks"}


def test_only_listed_answers_count_even_with_tolerance() -> None:
    marking = {**ACCEPTS_9_81, "with": {"answers": ["9.81", "10"], "tolerance": 0.02}}
    assert answer_decisions(marking) == {"9.81": "full_marks", "10": "full_marks"}
    cancelled = mark_by_example(marking, ["9.81"], "cancel")
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
    assert answer_decisions({"uses": method, "with": settings}) == {}


def test_cancellation_does_not_touch_other_questions_or_read_only_checks() -> None:
    children = [
        {**ACCEPTS_9_81, "question": "Q2"},
        {**ACCEPTS_9_81, "read_only": True},
        {**ACCEPTS_9_81, "value": 1},
    ]
    marking = {"question": "Q1", "uses": "gradeflow/combination@1", "with": {"checks": children}}
    assert answer_decisions(marking) == {}
    assert mark_by_example(marking, ["9.81"], "cancel") == marking


def test_recursive_cancellation_keeps_conditions_and_weights() -> None:
    prerequisite = {**ACCEPTS_9_81, "id": "prerequisite", "read_only": True}
    child = {**ACCEPTS_9_81, "if": "{{ checks.prerequisite.correct }}", "weight": 0.25}
    marking = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {"combine": "any", "checks": [prerequisite, child]},
    }
    assert answer_decisions(marking) == {"9.81": "part_marks"}
    cancelled = mark_by_example(marking, ["9.81"], "cancel")
    assert answer_decisions(cancelled) == {}
    remaining = cancelled["with"]["checks"][1]
    assert remaining["if"] == "{{ checks.prerequisite.correct }}"
    assert remaining["weight"] == 0.25


def test_double_negation_and_added_up_decisions() -> None:
    check: dict[str, Any] = {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["10"]}}
    for _ in range(2):
        check = {"uses": "gradeflow/not-allowed@1", "with": {"check": check}}
    assert answer_decisions(check) == {"10": "full_marks"}
    accepts_eleven = {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["11"]}}
    proportional: dict[str, Any] = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "proportional", "checks": [check, accepts_eleven]},
    }
    assert answer_decisions(proportional) == {"10": "part_marks", "11": "part_marks"}
    cancelled = mark_by_example(proportional, ["10"], "cancel")
    assert answer_decisions(cancelled) == {"11": "part_marks"}
    total_combination = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "total", "checks": [check]},
    }
    assert answer_decisions(total_combination, question=NumberQuestion(max_marks=1)) == {
        "10": "full_marks"
    }
    assert answer_decisions(total_combination, question=NumberQuestion(max_marks=2)) == {
        "10": "part_marks"
    }


@pytest.mark.parametrize(("weights", "max_marks"), [((0.01, 0.29, 0.7), 1), ((0.3, 1.4, 2.3), 4)])
def test_marks_adding_up_to_full_marks_are_full_despite_rounding(
    weights: tuple[float, ...], max_marks: float
) -> None:
    checks = [{**ACCEPTS_X, "weight": weight} for weight in weights]
    total_combination = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "total", "checks": checks},
    }
    question = TextQuestion(max_marks=max_marks)
    assert answer_decisions(total_combination, question=question) == {"x": "full_marks"}


def test_cancel_preserves_other_proportions_of_the_marks() -> None:
    marking = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "proportional",
            "checks": [
                {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["10"]}},
                {
                    "weight": 3,
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": ["9.81"]},
                },
            ],
        },
    }
    assert answer_proportions(marking) == [0.75, 0, 0, 0.25]
    cancelled = mark_by_example(marking, ["10"], "cancel")
    assert answer_proportions(cancelled) == [0.75, 0, 0, 0]
    assert answer_decisions(cancelled) == {"9.81": "part_marks"}


def test_an_incomplete_draft_has_no_decisions() -> None:
    for marking in (
        {},
        {"uses": "gradeflow/not-allowed@1", "with": {}},
        {"uses": "gradeflow/accepted-answers@1", "with": {}},
    ):
        assert answer_decisions(marking) == {}


def test_cancelling_an_absent_decision_changes_nothing() -> None:
    marking = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {"checks": [ACCEPTS_9_81]},
    }
    assert mark_by_example(marking, ["10"], "cancel") == marking
    cancelled = mark_by_example(marking, ["9.81"], "cancel")
    assert mark_by_example(cancelled, ["9.81"], "cancel") == cancelled


class AcceptedAnswersTwo(AcceptedAnswers):
    version = 2


def test_new_checks_use_the_latest_version_and_decisions_read_every_version() -> None:
    methods = Methods([*BUILTINS, AcceptedAnswersTwo()])
    marking = mark_by_example(ACCEPTS_9_81, ["10"], "full_marks", methods=methods)
    assert marking == {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "any",
            "checks": [
                {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["9.81"]}},
                {"uses": "gradeflow/accepted-answers@2", "with": {"answers": ["10"]}},
            ],
        },
    }
    assert answer_decisions(marking, methods=methods) == {"9.81": "full_marks", "10": "full_marks"}
    assert mark_by_example(marking, ["10"], "cancel", methods=methods) == ACCEPTS_9_81


def test_deciding_needs_the_methods_decisions_are_written_with() -> None:
    with pytest.raises(EditingError, match="There's no method gradeflow/accepted-answers"):
        answer_decisions(ACCEPTS_9_81, methods=Methods())
