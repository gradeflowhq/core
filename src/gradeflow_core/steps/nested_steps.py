"""Settings that hold steps: a check's own check, or the steps of a workflow, and how their
outcomes are combined."""

from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator, ValidationInfo
from pydantic_core import InitErrorDetails, PydanticCustomError, ValidationError

from ..common.hints import hint
from ..common.paths import location_of
from ..common.problems import InvalidError, Problem
from ..questions.types import QuestionId
from .combining import MARKS_BY_LABELS, MarksBy, Part
from .step import (
    METHODS_CONTEXT_KEY,
    STEP_PROPERTIES,
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
            "Step", [_error_details(problem, value) for problem in error.problems]
        ) from None


def _error_details(problem: Problem, value: Any) -> InitErrorDetails:
    """A problem of a step as pydantic reports one, where it is in the step's data."""
    is_setting = bool(problem.field) and problem.field[0] not in STEP_PROPERTIES
    return InitErrorDetails(
        type=PydanticCustomError("invalid_step", "{message}", {"message": problem.message}),
        loc=(*location_of(problem.step_path), *(("with",) if is_setting else ()), *problem.field),
        input=value,
    )


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
