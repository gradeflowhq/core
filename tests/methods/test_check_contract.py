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
from gradeflow_core.methods.code import runner as code_runner
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion


def evaluate(check: dict[str, Any], answer: str = "x", *, values: bool = False) -> Any:
    question = ValuesQuestion(value_types=["text", "text"]) if values else TextQuestion()
    questions = QuestionSet({"Q1": question})
    scheme = MarkScheme.load([{"question": "Q1", **check}])
    assert not scheme.problems(questions)
    return next(
        mark(questions, scheme, [Submission(student_id="s", answers={"Q1": answer})])
    ).results["Q1"]


GATE = {
    "id": "gate",
    "read_only": True,
    "uses": "gradeflow/not-allowed@1",
    "with": {"check": {"uses": "gradeflow/full-marks@1"}},
}
"""A check that is never correct, for conditions to read."""
SKIP, KEEP = "{{ checks.gate.correct }}", "{{ not checks.gate.correct }}"


def gated(check: dict[str, Any]) -> dict[str, Any]:
    return {"uses": "gradeflow/combination@1", "with": {"checks": [GATE, check]}}


@pytest.mark.parametrize(
    "method", ["gradeflow/full-marks@1", "gradeflow/not-allowed@1", "gradeflow/custom-code@1"]
)
def test_a_false_condition_precedes_method_execution(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("A skipped check executed custom code")

    monkeypatch.setattr(code_runner, "run", forbidden)
    settings = (
        {"check": {"uses": "gradeflow/full-marks@1"}}
        if method == "gradeflow/not-allowed@1"
        else {"code": "correct = True"}
        if method == "gradeflow/custom-code@1"
        else {}
    )
    result = evaluate(gated({"uses": method, "with": settings, "if": SKIP}))
    assert (result.proportion, result.feedback, result.marked) == (0, "", True)


@pytest.mark.parametrize("combine", ["all", "any", "none", "total", "proportional"])
@pytest.mark.parametrize("method", ["gradeflow/combination@1", "gradeflow/multiple-values@1"])
def test_skipped_children_are_omitted_even_when_all_skipped(combine: str, method: str) -> None:
    check = {
        "uses": method,
        "with": {
            "combine": combine,
            "checks": [
                {"uses": "gradeflow/full-marks@1", "if": SKIP},
                {"uses": "gradeflow/full-marks@1", "if": SKIP},
            ],
        },
    }
    result = evaluate(
        gated(check),
        "x, y" if method == "gradeflow/multiple-values@1" else "x",
        values=method == "gradeflow/multiple-values@1",
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
                {"uses": "gradeflow/full-marks@1", "if": SKIP},
                {"uses": "gradeflow/full-marks@1", "if": KEEP},
            ],
        },
    }
    assert evaluate(gated(check), "x, y", values=True).proportion == 1
    check["with"]["combine"] = "total"
    check["with"]["checks"][1]["weight"] = 0.5
    assert evaluate(gated(check), "x, y", values=True).proportion == 0.5


@pytest.mark.parametrize("method", ["gradeflow/combination@1", "gradeflow/multiple-values@1"])
def test_named_skips_are_available_to_later_conditions(method: str) -> None:
    check = {
        "uses": method,
        "with": {
            "checks": [
                {"id": "first", "uses": "gradeflow/full-marks@1", "if": SKIP},
                {"uses": "gradeflow/full-marks@1", "if": "{{ checks.first.skipped }}"},
            ]
        },
    }
    assert (
        evaluate(
            gated(check),
            "x, y" if method == "gradeflow/multiple-values@1" else "x",
            values=method == "gradeflow/multiple-values@1",
        ).proportion
        == 1
    )


def test_any_order_values_reject_sibling_scope() -> None:
    questions = QuestionSet({"Q1": ValuesQuestion(value_types=["text", "text"])})
    data = {
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
    scheme = MarkScheme.load([data])
    assert any("first" in problem.message for problem in scheme.problems(questions))


def test_wrapped_incompatible_check_rejected() -> None:
    library = BUILTINS.with_files(
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
    data = {"question": "Q1", "uses": "sample/wrapped@1"}
    problems = MarkScheme.load([data], methods=library).problems(questions)
    message = "Number Range can't mark a text question."
    assert [(p.check, p.field, p.message) for p in problems] == [("markings.0", (), message)]


def test_optional_conditional_branch_and_supplied_check_locations() -> None:
    questions = QuestionSet({"Q1": TextQuestion(), "Q2": TextQuestion()})
    data: dict[str, Any] = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [{"question": "Q1", "uses": "gradeflow/full-marks@1"}],
            "then": [{"question": "Q2", "uses": "gradeflow/full-marks@1"}],
        },
    }
    assert not MarkScheme.load([data]).problems(questions)
    data["with"]["then"] = [
        {"question": "Q2", "uses": "gradeflow/number-range@1", "with": {"min_value": 1}}
    ]
    [problem] = MarkScheme.load([data]).problems(questions)
    assert problem.check == "markings.0.with.then.0"


@pytest.mark.parametrize(
    "condition",
    ["{{ checks.later.correct }}", "{{ checks['later'].correct }}", "not an expression"],
)
def test_unavailable_or_invalid_root_conditions_report_validation_and_runtime_errors(
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
    assert not result.marked


def test_unmarked_indexed_prerequisite_is_an_error_not_a_skip() -> None:
    from gradeflow_core.questions.number import NumberQuestion

    questions = QuestionSet({"Q1": NumberQuestion(), "Q2": TextQuestion()})
    scheme = MarkScheme.load(
        [
            {
                "uses": "gradeflow/combination@1",
                "with": {
                    "checks": [
                        {
                            "id": "gate",
                            "question": "Q1",
                            "read_only": True,
                            "uses": "gradeflow/full-marks@1",
                        },
                        {
                            "question": "Q2",
                            "uses": "gradeflow/full-marks@1",
                            "if": "{{ checks['gate'].correct }}",
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
    assert not result.marked
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


def test_generated_loops_and_nested_compositions_keep_input_and_value_scope() -> None:
    library = BUILTINS.with_files(
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
    if: "{{ index == 1 and each == 'x' }}"
    with: {point: "{{ each }}"}
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
        methods=library,
    )
    assert not scheme.problems(questions)
    result = next(
        mark(questions, scheme, [Submission(student_id="s", answers={"Q1": "x, 2"})])
    ).results["Q1"]
    assert result.proportion == 1
    invalid = MarkScheme.load([{"question": "Q1", "uses": "sample/nested@1"}], methods=library)
    assert invalid.problems(QuestionSet({"Q1": TextQuestion()}))
