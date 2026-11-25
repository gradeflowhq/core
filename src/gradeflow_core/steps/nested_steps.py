"""Settings that hold steps: a check's own check, or the steps of a workflow, and how their
outcomes are combined."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator, ValidationInfo
from pydantic_core import InitErrorDetails, PydanticCustomError, ValidationError

from ..common.hints import hint
from ..common.problems import InvalidError, problem_error
from ..questions.types import QuestionId
from .combining import COMBINATIONS, MARKS_BY_LABELS, Combination, MarksBy, Part
from .outcome import CORRECT_FEEDBACK, Outcome, QuestionOutcomes
from .step import (
    METHODS_CONTEXT_KEY,
    Step,
    needs_workflow_problem,
    valid_step,
)


class StepData(BaseModel):
    """What a step looks like before its method is known, for its JSON schema."""

    model_config = ConfigDict(title="Step")

    uses: str
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")


def _validate_step(value: Any, info: ValidationInfo) -> Step:
    """A step as a setting holds it: loaded already, or still as written."""
    if isinstance(value, Step):
        return value
    try:
        return valid_step(value, (info.context or {})[METHODS_CONTEXT_KEY])
    except InvalidError as error:
        raise ValidationError.from_exception_data(
            "Step",
            [
                InitErrorDetails(type=problem_error(problem), input=value)
                for problem in error.problems
            ],
        ) from None


def _validate_check(value: Any, info: ValidationInfo) -> Step:
    step = _validate_step(value, info)
    if not step.resolution.gives_marks:
        message = needs_workflow_problem(step).message
        raise PydanticCustomError("gives_no_marks", "{message}", {"message": message})
    return step


_STEP_SERIALIZER = PlainSerializer(Step.dump, return_type=dict[str, Any])
NestedStep = Annotated[
    Step,
    PlainValidator(_validate_step, json_schema_input_type=StepData),
    _STEP_SERIALIZER,
    Field(title="Step", json_schema_extra=hint(kind="step")),
]
"""A step a setting holds, a check or a transform."""
NestedCheck = Annotated[
    Step,
    PlainValidator(_validate_check, json_schema_input_type=StepData),
    _STEP_SERIALIZER,
    Field(title="Check", json_schema_extra=hint(kind="step")),
]
"""A step a setting holds that must give marks."""
MarksBySetting = Annotated[
    MarksBy, Field(title="How marks are given", json_schema_extra=hint(labels=MARKS_BY_LABELS))
]
"""How the checks a setting holds are combined."""


def combination_of(settings: Mapping[str, Any]) -> Combination | None:
    """How ``settings``, as YAML holds them, combine the checks they hold; None if their
    ``marks_by`` is none of the ways."""
    marks_by = settings.get("marks_by", "all_correct")
    return next(
        (combination for name, combination in COMBINATIONS.items() if name == marks_by), None
    )


def combined_part(check: Step, default_name: str | None = None) -> Part:
    """``check`` as one of those whose outcomes are combined, called ``default_name`` in feedback
    if it has no name."""
    return Part(
        check.weight,
        check.name,
        default_name,
        check.method_label,
        bool(check.feedback),
        check.describe,
    )


def questions_marked_by(steps: Sequence[Step], question_id: QuestionId | None) -> set[QuestionId]:
    """The questions the steps that count give marks to."""
    return {
        marked_question_id
        for step in steps
        if step.weight != 0
        for marked_question_id in step.questions_marked(question_id)
    }


def questions_read_by(steps: Sequence[Step], question_id: QuestionId | None) -> set[QuestionId]:
    """The questions whose answers the steps read."""
    return {
        read_question_id for step in steps for read_question_id in step.questions_read(question_id)
    }


def labelled_combined_part(
    check: Step, outcomes: QuestionOutcomes, default_name: str | None
) -> tuple[Part, QuestionOutcomes]:
    """A check and its outcomes to combine. One without a name is called ``default_name``, the
    part or item it marked, so each line of feedback says which it is."""
    part = combined_part(check, default_name)
    if check.name or default_name is None:
        return part, outcomes
    return part, {
        question_id: replace(outcome, labels=(*outcome.labels, default_name))
        if isinstance(outcome, Outcome)
        else outcome
        for question_id, outcome in outcomes.items()
    }


def with_correct_feedback(outcomes: QuestionOutcomes) -> QuestionOutcomes:
    """An answer whose every part or item is correct is “Correct.”, without a line for each."""
    return {
        question_id: replace(outcome, feedback=CORRECT_FEEDBACK)
        if isinstance(outcome, Outcome) and outcome.correct
        else outcome
        for question_id, outcome in outcomes.items()
    }
