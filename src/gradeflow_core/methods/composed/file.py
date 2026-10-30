from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ...common.hints import hint
from ...common.paths import Loc
from ...questions import TYPE_LABELS
from ...questions.types import QuestionType
from ..example import Example
from ..logic.combination import Combination
from ..method import Level

COMBINATION = Combination().ref
"""What a method file's checks and ``combine`` make up."""
BODY = ("checks", "combine")
"""What of a method file is the settings of its Combination."""
TEMPLATE_KEYS = frozenset({"if", "feedback"})
"""Keys of a check in a method file that hold templates, rendered when it marks."""


class ItemField(BaseModel):
    """A field of each item of a list a repeated check goes through, as the input names it."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    description: str | None = None
    examples: list[JsonValue] | None = None


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    description: str | None = None
    default: JsonValue = None
    """Without an explicit default, the caller must supply the input. A list needs an item
    unless its default is empty."""
    advanced: bool | None = None
    """Whether most people leave it at its default; by default, as for the setting it fills."""
    choices: list[str] | None = None
    """Fewer choices than the setting it fills allows."""
    labels: dict[str, str] | None = None
    applies_when: dict[str, JsonValue] | None = None
    """The input applies only when other inputs have these values, as a setting's hint says."""
    examples: list[JsonValue] | None = None
    items: dict[str, ItemField] | None = None
    """For a list a repeated check goes through: the fields ``each.…`` names, titled for it
    rather than as the settings they fill."""


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
        default=[], title="Answer types", json_schema_extra=hint(labels=TYPE_LABELS)
    )
    group: bool = False
    inputs: dict[str, Input] = {}
    combine: JsonValue = "all"
    checks: list[JsonValue] | str
    description: str | None = Field(
        default=None,
        title="Summary template",
        description="Replaces the summary its checks write.",
        json_schema_extra=hint(kind="markdown", advanced=True),
    )
    feedback: str | None = Field(
        default=None,
        title="Feedback template",
        description="Replaces the feedback its checks give.",
        json_schema_extra=hint(kind="markdown", advanced=True),
    )
    examples: list[Example] = []

    def body(self) -> dict[str, Any]:
        """The Combination the method is, with the inputs not yet filled in."""
        return to_body(self.model_dump(include=set(BODY)))


def to_body(file: Mapping[str, Any]) -> dict[str, Any]:
    """A method file's checks as the Combination of them."""
    return {"uses": COMBINATION, "with": {key: deepcopy(file[key]) for key in BODY if key in file}}


def to_file_body(check: Mapping[str, Any]) -> dict[str, Any]:
    """A check as a method file's checks: a Combination's own, or else the check alone."""
    if check.get("uses") == COMBINATION and check.keys() <= {"uses", "with"}:
        return {key: value for key, value in check.get("with", {}).items() if key in BODY}
    return {"checks": [check]}


def to_file_path(path: Loc) -> Loc:
    """Where a path in the body is in the file: ``with.checks`` is ``checks``."""
    return path[1:] if path[:1] == ("with",) else path


class Fault(ValueError):
    def __init__(self, message: str, path: Loc) -> None:
        super().__init__(message)
        self.path = path


def methods_used(data: Any) -> set[str]:
    """The method references at any depth in a marking or method file, as YAML holds it."""
    if isinstance(data, Mapping):
        own = {data["uses"]} if isinstance(data.get("uses"), str) else set()
        return own.union(*(methods_used(value) for value in data.values()))
    if isinstance(data, list):
        return set().union(*(methods_used(item) for item in data))
    return set()
