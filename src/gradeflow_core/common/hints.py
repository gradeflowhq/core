from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue
from pydantic.fields import FieldInfo

from .schemas import JsonDict

HINT_KEY = "x-gradeflow"


class Hint(BaseModel):
    """What a field holds, beyond its JSON schema, under the ``x-gradeflow`` key of its property."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["check", "code", "markdown", "question", "feedback", "condition"] | None = None
    """What kind of value the field holds. "markdown" is Markdown with LaTeX formulas between $
    signs. A list of checks says "check" on its items."""
    language: Literal["python"] | None = None
    advanced: bool = False
    """An option most people leave at its default."""
    labels: dict[str, str] | None = None
    """Plain labels for enum values."""
    applies_when: dict[str, JsonValue] | None = None
    """The field applies only when sibling fields of the same object have these values; a list
    means any of them."""
    accepts_answers: bool = False
    """The field lists accepted or model answers."""
    item: str | None = None
    """What one entry of a list or map field is called, such as "parameter"."""
    names: bool = False
    """The field's value names the object holding it."""
    inserts: dict[str, str] | None = None
    """What a template may include, by label."""

    def schema_extra(self) -> JsonDict:
        """The hint as a field's ``json_schema_extra``, without its defaults."""
        return {HINT_KEY: self.model_dump(exclude_defaults=True)}


def hint(**values: Any) -> JsonDict:
    """A field's ``json_schema_extra`` with a hint of ``values``."""
    return Hint(**values).schema_extra()


def hint_of(field: FieldInfo) -> Hint:
    """The hint in a field's ``json_schema_extra``; an empty one if it has none."""
    extra = field.json_schema_extra
    found = extra.get(HINT_KEY) if isinstance(extra, Mapping) else None
    return Hint.model_validate(found or {})


ADVANCED = hint(advanced=True)
CODE: dict[str, Any] = {"kind": "code", "language": "python"}
"""The hint values of Python code, for ``hint(**CODE)``."""
