from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue
from pydantic.fields import FieldInfo

from .schemas import JsonDict

HINT_KEY = "x-gradeflow"


class Hint(BaseModel):
    """What a field holds, beyond its JSON schema, under the ``x-gradeflow`` key of its property."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: (
        Literal["step", "output_step", "code", "markdown", "question", "feedback", "condition"]
        | None
    ) = None
    """What kind of value the field holds. "markdown" is Markdown with LaTeX formulas between $
    signs. A list of steps says "step" on its items."""
    language: Literal["python"] | None = None
    advanced: bool = False
    """An option most people leave at its default."""
    labels: dict[str, str] | None = None
    """Plain labels for enum values."""
    applies_when: dict[str, list[JsonValue]] | None = None
    """The field applies only when each named sibling field of the same object has one of the
    values listed for it."""
    lists_answers: bool = False
    """The field lists accepted or model answers."""
    item_label: str | None = None
    """What one entry of a list or map field is called, such as "parameter"."""
    is_name: bool = False
    """The field's value names the object holding it."""
    template_parts: dict[str, str] | None = None
    """What a template may include, by label."""

    def schema_extra(self) -> JsonDict:
        """The hint as a field's ``json_schema_extra``, without its defaults."""
        return {HINT_KEY: self.model_dump(exclude_defaults=True)}


def _as_schema_extra[**P](make_hint: Callable[P, Hint]) -> Callable[P, JsonDict]:
    def schema_extra(*args: P.args, **kwargs: P.kwargs) -> JsonDict:
        return make_hint(*args, **kwargs).schema_extra()

    return schema_extra


hint = _as_schema_extra(Hint)
"""A field's ``json_schema_extra`` with a hint of these values."""


def hint_of(field: FieldInfo) -> Hint:
    """The hint in a field's ``json_schema_extra``; an empty one if it has none."""
    extra = field.json_schema_extra
    hint_values = extra.get(HINT_KEY) if isinstance(extra, Mapping) else None
    return Hint.model_validate(hint_values or {})


ADVANCED_HINT = hint(advanced=True)
PYTHON_CODE_HINT: dict[str, Any] = {"kind": "code", "language": "python"}
"""The hint values of Python code, for ``hint(**PYTHON_CODE_HINT)``."""
