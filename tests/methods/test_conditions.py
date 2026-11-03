from typing import Any

import pytest

from gradeflow_core import (
    BUILTINS,
    MarkScheme,
    MethodFileError,
    QuestionSet,
    Submission,
    mark,
)
from gradeflow_core.methods.code import runner
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion


def mark_q1(check: dict[str, Any], answer: str = "x", *, two_values: bool = False) -> Any:
    question = ValuesQuestion(value_types=["text", "text"]) if two_values else TextQuestion()
    questions = QuestionSet({"Q1": question})
    scheme = MarkScheme.load([{"question": "Q1", **check}])
    assert not scheme.problems(questions)
    return next(
        mark(questions, scheme, [Submission(student_id="s", answers={"Q1": answer})])
    ).results["Q1"]


NEVER_CORRECT = {
    "id": "never_correct",
    "read_only": True,
    "uses": "gradeflow/not-allowed@1",
    "with": {"check": {"uses": "gradeflow/full-marks@1"}},
}
"""A check that is never correct, for conditions to read."""
ALWAYS_FALSE, ALWAYS_TRUE = (
    "{{ checks.never_correct.correct }}",
    "{{ not checks.never_correct.correct }}",
)


def after_never_correct(check: dict[str, Any]) -> dict[str, Any]:
    return {"uses": "gradeflow/combination@1", "with": {"checks": [NEVER_CORRECT, check]}}


@pytest.mark.parametrize(
    "method", ["gradeflow/full-marks@1", "gradeflow/not-allowed@1", "gradeflow/custom-code@1"]
)
def test_a_false_condition_precedes_method_execution(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("A skipped check executed custom code")

    monkeypatch.setattr(runner, "run", forbidden)
    settings = (
        {"check": {"uses": "gradeflow/full-marks@1"}}
        if method == "gradeflow/not-allowed@1"
        else {"code": "correct = True"}
        if method == "gradeflow/custom-code@1"
        else {}
    )
    result = mark_q1(after_never_correct({"uses": method, "with": settings, "if": ALWAYS_FALSE}))
    assert (result.proportion, result.feedback, result.is_marked) == (0, "", True)


@pytest.mark.parametrize("combine", ["all", "any", "none", "total", "proportional"])
@pytest.mark.parametrize("method", ["gradeflow/combination@1", "gradeflow/multiple-values@1"])
def test_skipped_checks_count_for_nothing_even_when_all_are_skipped(
    combine: str, method: str
) -> None:
    check = {
        "uses": method,
        "with": {
            "combine": combine,
            "checks": [
                {"uses": "gradeflow/full-marks@1", "if": ALWAYS_FALSE},
                {"uses": "gradeflow/full-marks@1", "if": ALWAYS_FALSE},
            ],
        },
    }
    result = mark_q1(
        after_never_correct(check),
        "x, y" if method == "gradeflow/multiple-values@1" else "x",
        two_values=method == "gradeflow/multiple-values@1",
    )
    assert result.proportion == 0


@pytest.mark.parametrize("any_order", [False, True])
def test_skipped_values_count_for_nothing(any_order: bool) -> None:
    check: dict[str, Any] = {
        "uses": "gradeflow/multiple-values@1",
        "with": {
            "combine": "proportional",
            "any_order": any_order,
            "checks": [
                {"uses": "gradeflow/full-marks@1", "if": ALWAYS_FALSE},
                {"uses": "gradeflow/full-marks@1", "if": ALWAYS_TRUE},
            ],
        },
    }
    assert mark_q1(after_never_correct(check), "x, y", two_values=True).proportion == 1
    check["with"]["combine"] = "total"
    check["with"]["checks"][1]["weight"] = 0.5
    assert mark_q1(after_never_correct(check), "x, y", two_values=True).proportion == 0.5


@pytest.mark.parametrize("method", ["gradeflow/combination@1", "gradeflow/multiple-values@1"])
def test_named_skips_are_available_to_later_conditions(method: str) -> None:
    check = {
        "uses": method,
        "with": {
            "checks": [
                {"id": "first", "uses": "gradeflow/full-marks@1", "if": ALWAYS_FALSE},
                {"uses": "gradeflow/full-marks@1", "if": "{{ checks.first.skipped }}"},
            ]
        },
    }
    assert (
        mark_q1(
            after_never_correct(check),
            "x, y" if method == "gradeflow/multiple-values@1" else "x",
            two_values=method == "gradeflow/multiple-values@1",
        ).proportion
        == 1
    )


def test_any_order_values_cant_read_earlier_checks() -> None:
    questions = QuestionSet({"Q1": ValuesQuestion(value_types=["text", "text"])})
    marking = {
        "question": "Q1",
        "uses": "gradeflow/multiple-values@1",
        "with": {
            "any_order": True,
            "checks": [
                {"id": "first", "uses": "gradeflow/full-marks@1"},
                {"uses": "gradeflow/full-marks@1", "if": "{{ checks.first.correct }}"},
            ],
        },
    }
    scheme = MarkScheme.load([marking])
    assert any("first" in problem.message for problem in scheme.problems(questions))


def test_wrapped_incompatible_check_rejected() -> None:
    methods = BUILTINS.with_files(
        [
            (
                "sample/wrapped",
                """label: Wrapped
version: 1
summary: A wrapped number range.
question_types: [text]
checks:
  - uses: gradeflow/number-range@1
    with: {min_value: 1}
""",
            )
        ]
    )
    questions = QuestionSet({"Q1": TextQuestion()})
    marking = {"question": "Q1", "uses": "sample/wrapped@1"}
    problems = MarkScheme.load([marking], methods=methods).problems(questions)
    message = "Number Range can't mark a text question."
    assert [(problem.check, problem.field, problem.message) for problem in problems] == [
        ("markings.0", (), message)
    ]


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
    assert problem.check == "markings.0.with.then.0"


@pytest.mark.parametrize(
    "condition",
    ["{{ checks.later.correct }}", "{{ checks['later'].correct }}", "not an expression"],
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
                "uses": "gradeflow/combination@1",
                "with": {
                    "checks": [
                        {
                            "id": "q1",
                            "question": "Q1",
                            "read_only": True,
                            "uses": "gradeflow/full-marks@1",
                        },
                        {
                            "question": "Q2",
                            "uses": "gradeflow/full-marks@1",
                            "if": "{{ checks['q1'].correct }}",
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


def test_inputs_cannot_bind_settings_with_incompatible_types() -> None:
    with pytest.raises(MethodFileError, match="different types"):
        BUILTINS.with_files(
            [
                (
                    "sample/conflicting",
                    """label: Conflicting
version: 1
summary: One input fills settings with incompatible types.
question_types: [text]
inputs:
  threshold: {}
checks:
  - uses: gradeflow/key-point@1
    with: {point: "{{ inputs.threshold }}"}
  - uses: gradeflow/number-range@1
    with: {min_value: "{{ inputs.threshold }}"}
""",
                )
            ]
        )


def test_repeated_and_nested_method_file_checks_read_their_own_inputs_and_value() -> None:
    methods = BUILTINS.with_files(
        [
            (
                "sample/range",
                """label: Range
version: 1
summary: A numeric range.
question_types: [number, text]
checks:
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
checks:
  - uses: sample/range@1
""",
            ),
            (
                "sample/loop",
                """label: Loop
version: 1
summary: Input and loop scope.
question_types: [text]
inputs:
  words: {}
checks:
  - for_each: "{{ inputs.words }}"
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
                    "checks": [
                        {"uses": "sample/loop@1", "with": {"words": ["x", "y"]}},
                        {"uses": "sample/nested@1"},
                    ]
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
