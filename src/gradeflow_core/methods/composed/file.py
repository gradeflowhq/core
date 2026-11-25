from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ...common.hints import hint
from ...common.problems import InvalidError, Problem
from ...questions import QUESTION_CLASSES
from ...questions.types import QuestionType
from ...steps.example import Example
from ...steps.method import Level
from ...steps.workflow import Workflow

WORKFLOW_REF = Workflow().ref
WORKFLOW_KEYS = ("steps", "marks_by", "output", "output_step")
"""The keys of a method file that are its workflow's settings."""
TEMPLATE_KEYS = frozenset({"if", "feedback", "input", "output_step"})
"""Keys of a step, or of its settings, in a method file that hold templates, read when it runs."""


class DeclaredItemField(BaseModel):
    """How a declared setting titles a field of the items a repeated step goes through."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    description: str | None = None


SettingType = Literal["number", "integer", "text", "boolean", "text_list", "number_list"]
SETTING_TYPE_LABELS: dict[str, str] = {
    "number": "Number",
    "integer": "Whole number",
    "text": "Text",
    "boolean": "Yes/No",
    "text_list": "List of text",
    "number_list": "List of numbers",
}


class DeclaredSetting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    type: SettingType | None = Field(
        default=None,
        title="Type",
        json_schema_extra=hint(labels=SETTING_TYPE_LABELS),
    )
    description: str | None = None
    default: JsonValue = None
    """Without an explicit default, the caller must fill it in. A list needs an item
    unless its default is empty."""
    advanced: bool | None = None
    """Whether most people leave it at its default; by default, as for the setting it fills."""
    choices: list[str] | list[int | float] | None = None
    """The values it may have, of its type or among the options of the setting it fills."""
    labels: dict[str, str] | None = None
    applies_when: dict[str, list[JsonValue]] | None = None
    """It applies only when other settings have these values."""
    items: dict[str, DeclaredItemField] | None = None
    """For a list a repeated step goes through: the fields ``item.…`` names, titled for it
    rather than as the settings they fill."""

    @field_validator("choices")
    @classmethod
    def _choices_of_its_type(
        cls, choices: list[str] | list[int | float] | None, info: ValidationInfo
    ) -> list[str] | list[int | float] | None:
        setting_type = info.data.get("type")
        if not choices or setting_type is None:
            return choices
        match setting_type:
            case "text" if not all(isinstance(choice, str) for choice in choices):
                raise ValueError("Each choice must be text.")
            case "number" if any(isinstance(choice, str) for choice in choices):
                raise ValueError("Each choice must be a number.")
            case "integer" if not all(isinstance(choice, int) for choice in choices):
                raise ValueError("Each choice must be a whole number.")
            case "boolean" | "text_list" | "number_list":
                raise ValueError(f"A {SETTING_TYPE_LABELS[setting_type]} setting has no choices.")
        return choices


class MethodFile(BaseModel):
    """What a composed method's YAML holds."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(title="Name")
    version: int = Field(gt=0, strict=True)
    summary: str = Field(
        default="", title="Summary", description="What earns the marks, in 10 words or fewer."
    )
    guide: str = Field(
        default="",
        title="Guide",
        description="Use it when, How it marks, Good to know and Related, as built-in methods "
        "write them.",
        json_schema_extra=hint(kind="markdown"),
    )
    level: Level = Field(
        default="basic",
        title="Level",
        description="Basic methods suit most questions, advanced ones particular cases.",
        json_schema_extra=hint(
            advanced=True,
            labels={"basic": "Basic", "advanced": "Advanced", "internal": "Inside other methods"},
        ),
    )
    question_types: list[QuestionType] | Literal["all"] = Field(
        default=[],
        title="Question types",
        json_schema_extra=hint(
            labels={question_type: cls.label for question_type, cls in QUESTION_CLASSES.items()}
        ),
    )
    marks_question_group: bool = False
    settings: dict[str, DeclaredSetting] = {}
    steps: list[JsonValue] | str
    output: Literal["marks", "step_output"] = "marks"
    """Marks, or the output of ``output_step``."""
    marks_by: JsonValue = "all_correct"
    output_step: str | None = None
    description: str | None = Field(
        default=None,
        title="Description template",
        description="Replaces the description its steps write.",
        json_schema_extra=hint(kind="markdown", advanced=True),
    )
    feedback: str | None = Field(
        default=None,
        title="Feedback template",
        description="Replaces the feedback its steps give.",
        json_schema_extra=hint(kind="markdown", advanced=True),
    )
    examples: list[Example] = []

    @model_validator(mode="after")
    def _validate_output_step(self) -> Self:
        if self.output == "step_output" and self.output_step is None:
            raise ValueError("Name the step whose output it gives.")
        return self

    def workflow_step_data(self) -> dict[str, Any]:
        """The workflow the method is, with its settings not yet filled in."""
        return as_workflow(self.model_dump(include=set(WORKFLOW_KEYS)))


def as_workflow(file: Mapping[str, Any]) -> dict[str, Any]:
    """A method file's steps as a workflow step."""
    settings = {key: deepcopy(file[key]) for key in WORKFLOW_KEYS if file.get(key) is not None}
    return {"uses": WORKFLOW_REF, "with": settings}


def as_file_steps(step_data: Mapping[str, Any]) -> dict[str, Any]:
    """A step as a method file's steps: a workflow's own, or else the step alone."""
    if step_data.get("uses") == WORKFLOW_REF and step_data.keys() <= {"uses", "with"}:
        return {
            key: value for key, value in step_data.get("with", {}).items() if key in WORKFLOW_KEYS
        }
    return {"steps": [step_data]}


def method_file_error(message: str, path: str) -> InvalidError:
    """What is wrong at the JSON Pointer ``path`` in a method file."""
    return InvalidError([Problem(message, field=path)])


def methods_used(yaml_data: Any) -> set[str]:
    """The method references at any depth in a marking or method file, as YAML holds it."""
    if isinstance(yaml_data, Mapping):
        used_here = {yaml_data["uses"]} if isinstance(yaml_data.get("uses"), str) else set()
        return used_here.union(*(methods_used(value) for value in yaml_data.values()))
    if isinstance(yaml_data, list):
        return set().union(*(methods_used(item) for item in yaml_data))
    return set()


def methods_used_by_file(file: MethodFile) -> set[str]:
    """Composition references, including defaults; examples are evaluated separately."""
    return methods_used(file.model_dump(exclude={"examples"}))
