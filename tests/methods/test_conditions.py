from typing import Any
from unittest.mock import Mock

import pytest

from gradeflow_core import (
    BUILTINS,
    MarkScheme,
    MethodFileError,
    Question,
    QuestionSet,
    Submission,
    mark,
)
from gradeflow_core.methods.code import runner
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

from . import helpers
from .helpers import WRAPPED_NUMBER_RANGE, with_value_inputs

NEVER_CORRECT = {
    "id": "never_correct",
    "weight": 0,
    "uses": "gradeflow/not-allowed@1",
    "with": {"check": {"uses": "gradeflow/full-marks@1"}},
}
"""A check that is never correct, for conditions to read."""
ALWAYS_FALSE, ALWAYS_TRUE = (
    "{{ steps.never_correct.output.correct }}",
    "{{ not steps.never_correct.output.correct }}",
)


TWO_VALUES = ValuesQuestion(value_types=["text", "text"])
METHODS_OF_CHECKS = pytest.mark.parametrize(
    ("method", "answer", "question"),
    [("gradeflow/workflow@1", "x", None), ("gradeflow/multiple-values@1", "x, y", TWO_VALUES)],
)
"""Methods holding checks, with an answer for each to mark."""


def after_never_correct(check: dict[str, Any]) -> dict[str, Any]:
    return {"uses": "gradeflow/workflow@1", "with": {"steps": [NEVER_CORRECT, check]}}


def check_holding(method: str, checks: list[dict[str, Any]], **settings: Any) -> dict[str, Any]:
    checks_setting = (
        {"steps": checks}
        if method == "gradeflow/workflow@1"
        else {"checks": with_value_inputs(*checks)}
    )
    return {"uses": method, "with": {**settings, **checks_setting}}


@pytest.mark.parametrize(
    ("method", "settings"),
    [
        ("gradeflow/full-marks@1", {}),
        ("gradeflow/not-allowed@1", {"check": {"uses": "gradeflow/full-marks@1"}}),
        ("gradeflow/custom-code@1", {"code": "correct = True"}),
    ],
)
def test_a_false_condition_precedes_method_execution(
    monkeypatch: pytest.MonkeyPatch, method: str, settings: dict[str, Any]
) -> None:
    run = Mock()
    monkeypatch.setattr(runner, "run", run)
    result = helpers.mark(
        after_never_correct({"uses": method, "with": settings, "if": ALWAYS_FALSE}), "x"
    )
    assert (result.proportion, result.feedback, result.is_marked) == (0, "", True)
    run.assert_not_called()


@pytest.mark.parametrize(
    "marks_by", ["all_correct", "any_correct", "none_correct", "total", "proportional"]
)
@METHODS_OF_CHECKS
def test_skipped_checks_count_for_nothing_even_when_all_are_skipped(
    marks_by: str, method: str, answer: str, question: Question | None
) -> None:
    check = check_holding(
        method,
        [
            {"uses": "gradeflow/full-marks@1", "if": ALWAYS_FALSE},
            {"uses": "gradeflow/full-marks@1", "if": ALWAYS_FALSE},
        ],
        marks_by=marks_by,
    )
    result = helpers.mark(after_never_correct(check), answer, question)
    assert (result.proportion, result.is_marked) == (0, True)


@pytest.mark.parametrize("any_order", [False, True])
def test_skipped_values_count_for_nothing(any_order: bool) -> None:
    check: dict[str, Any] = {
        "uses": "gradeflow/multiple-values@1",
        "with": {
            "marks_by": "proportional",
            "any_order": any_order,
            "checks": with_value_inputs(
                {"uses": "gradeflow/full-marks@1", "if": ALWAYS_FALSE},
                {"uses": "gradeflow/full-marks@1", "if": ALWAYS_TRUE},
            ),
        },
    }
    assert helpers.mark(after_never_correct(check), "x, y", TWO_VALUES).proportion == 1
    check["with"]["marks_by"] = "total"
    check["with"]["checks"][1]["weight"] = 0.5
    assert helpers.mark(after_never_correct(check), "x, y", TWO_VALUES).proportion == 0.5


@METHODS_OF_CHECKS
def test_named_skips_are_available_to_later_conditions(
    method: str, answer: str, question: Question | None
) -> None:
    check = check_holding(
        method,
        [
            {"id": "first", "uses": "gradeflow/full-marks@1", "if": ALWAYS_FALSE},
            {"uses": "gradeflow/full-marks@1", "if": "{{ steps.first.skipped }}"},
        ],
    )
    assert helpers.mark(after_never_correct(check), answer, question).proportion == 1


def test_any_order_values_cant_read_earlier_checks() -> None:
    questions = QuestionSet({"Q1": ValuesQuestion(value_types=["text", "text"])})
    marking = {
        "question": "Q1",
        "uses": "gradeflow/multiple-values@1",
        "with": {
            "any_order": True,
            "checks": with_value_inputs(
                {"id": "first", "uses": "gradeflow/full-marks@1"},
                {"uses": "gradeflow/full-marks@1", "if": "{{ steps.first.output.correct }}"},
            ),
        },
    }
    scheme = MarkScheme.load([marking])
    assert any("first" in problem.message for problem in scheme.problems(questions))


def test_wrapped_incompatible_check_rejected() -> None:
    methods = BUILTINS.with_files([WRAPPED_NUMBER_RANGE])
    questions = QuestionSet({"Q1": TextQuestion()})
    marking = {"question": "Q1", "uses": "sample/wrapped@1"}
    problems = MarkScheme.load([marking], methods=methods).problems(questions)
    message = "Number Range can't be used on a text question."
    assert helpers.located(problems) == [("markings.0", (), message)]


def test_problems_in_a_conditionals_branch_name_their_path() -> None:
    questions = QuestionSet({"Q1": TextQuestion(), "Q2": TextQuestion()})
    marking: dict[str, Any] = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [{"question": "Q1", "uses": "gradeflow/full-marks@1"}],
            "then": [{"question": "Q2", "uses": "gradeflow/full-marks@1"}],
        },
    }
    assert not MarkScheme.load([marking]).problems(questions)
    marking["with"]["then"] = [
        {"question": "Q2", "uses": "gradeflow/number-range@1", "with": {"min_value": 1}}
    ]
    [problem] = MarkScheme.load([marking]).problems(questions)
    assert problem.step_path == "markings.0.with.then.0"


@pytest.mark.parametrize(
    "condition",
    [
        "{{ steps.later.output.correct }}",
        "{{ steps['later'].output.correct }}",
        "not an expression",
    ],
)
def test_a_top_level_condition_it_cant_read_is_a_problem_and_leaves_answers_unmarked(
    condition: str,
) -> None:
    questions = QuestionSet({"Q1": TextQuestion()})
    scheme = MarkScheme.load(
        [{"question": "Q1", "uses": "gradeflow/full-marks@1", "if": condition}]
    )
    assert scheme.problems(questions)
    result = next(
        mark(questions, scheme, [Submission(student_id="s", answers={"Q1": ""})])
    ).results["Q1"]
    assert not result.is_marked


def test_a_condition_on_an_unmarked_check_leaves_the_answer_unmarked() -> None:
    from gradeflow_core.questions.number import NumberQuestion

    questions = QuestionSet({"Q1": NumberQuestion(), "Q2": TextQuestion()})
    scheme = MarkScheme.load(
        [
            {
                "uses": "gradeflow/workflow@1",
                "with": {
                    "steps": [
                        {
                            "id": "q1",
                            "question": "Q1",
                            "weight": 0,
                            "uses": "gradeflow/full-marks@1",
                        },
                        {
                            "question": "Q2",
                            "uses": "gradeflow/full-marks@1",
                            "if": "{{ steps['q1'].output.correct }}",
                        },
                    ]
                },
            }
        ]
    )
    assert not scheme.problems(questions)
    result = next(
        mark(
            questions, scheme, [Submission(student_id="s", answers={"Q1": "unreadable", "Q2": "x"})]
        )
    ).results["Q2"]
    assert not result.is_marked
    assert "read" in result.feedback


def test_settings_cannot_bind_fields_with_incompatible_types() -> None:
    with pytest.raises(MethodFileError, match="different types"):
        BUILTINS.with_files(
            [
                (
                    "sample/conflicting",
                    """label: Conflicting
version: 1
summary: One setting fills fields with incompatible types.
question_types: [text]
settings:
  threshold: {}
steps:
  - uses: gradeflow/key-point@1
    with: {point: "{{ settings.threshold }}"}
  - uses: gradeflow/number-range@1
    with: {min_value: "{{ settings.threshold }}"}
""",
                )
            ]
        )


def test_repeated_and_nested_method_file_checks_read_their_own_settings_and_value() -> None:
    methods = BUILTINS.with_files(
        [
            (
                "sample/range",
                """label: Range
version: 1
summary: A numeric range.
question_types: [number, text]
steps:
  - uses: gradeflow/number-range@1
    with: {min_value: 1}
""",
            ),
            (
                "sample/nested",
                """label: Nested
version: 1
summary: Wrap a range again.
question_types: [number, text]
steps:
  - uses: sample/range@1
""",
            ),
            (
                "sample/loop",
                """label: Loop
version: 1
summary: Setting and loop scope.
question_types: [text]
settings:
  words: {}
steps:
  - for_each: "{{ settings.words }}"
    uses: gradeflow/key-point@1
    if: "{{ index == 1 and item == 'x' }}"
    with: {point: "{{ item }}"}
""",
            ),
        ]
    )
    questions = QuestionSet({"Q1": ValuesQuestion(value_types=["text", "number"])})
    scheme = MarkScheme.load(
        [
            {
                "question": "Q1",
                "uses": "gradeflow/multiple-values@1",
                "with": {
                    "checks": with_value_inputs(
                        {"uses": "sample/loop@1", "with": {"words": ["x", "y"]}},
                        {"uses": "sample/nested@1"},
                    )
                },
            }
        ],
        methods=methods,
    )
    assert not scheme.problems(questions)
    result = next(
        mark(questions, scheme, [Submission(student_id="s", answers={"Q1": "x, 2"})])
    ).results["Q1"]
    assert result.proportion == 1
    invalid = MarkScheme.load([{"question": "Q1", "uses": "sample/nested@1"}], methods=methods)
    assert invalid.problems(QuestionSet({"Q1": TextQuestion()}))
