"""A method's settings, and the models made from them for a place in a marking."""

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal, get_args, get_origin

from pydantic import AfterValidator, BaseModel, ConfigDict, create_model
from pydantic.fields import FieldInfo

IDENTIFIER = r"^[A-Za-z_][A-Za-z0-9_]*$"
"""A name code and conditions can use: letters, digits and underscores, not starting with a
digit."""


class Settings(BaseModel):
    """A method's settings: what a check's ``with`` holds."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("Can't be blank.")
    return text


Text = Annotated[str, AfterValidator(_not_blank)]
"""Text a check marks with, such as an accepted answer, which can't be only spaces."""


def limits_in_order(low: float | None, high: float | None, message: str) -> None:
    """Raises ``ValueError`` with ``message`` if both limits are set and ``high`` is below."""
    if low is not None and high is not None and high < low:
        raise ValueError(message)


def literal(values: Iterable[Any]) -> Any:
    return Literal.__getitem__(tuple(values))


def annotated(annotation: Any, *metadata: Any) -> Any:
    """``annotation`` with ``metadata``, such as a field's constraints."""
    return Annotated.__getitem__((annotation, *metadata)) if metadata else annotation


def item_field(info: FieldInfo) -> FieldInfo | None:
    """The field of each item of a list setting; None for any other setting."""
    if get_origin(info.annotation) is not list:
        return None
    return FieldInfo.from_annotation(get_args(info.annotation)[0])


def specialised(
    settings: type[Settings], *, drop: Iterable[str] = (), **changes: Mapping[str, Any]
) -> type[Settings]:
    """``settings`` narrowed to a place in a marking: without the fields in ``drop``, and with
    ``changes``, such as a suggestion hint or narrower type, merged into others."""
    fields: dict[str, Any] = {}
    for name, info in settings.model_fields.items():
        if name not in drop:
            merged = _changed(info, changes.get(name, {}))
            fields[name] = (merged.annotation, merged)
    return create_model(
        settings.model_config.get("title") or settings.__name__, __base__=Settings, **fields
    )


def _changed(info: FieldInfo, changes: Mapping[str, Any]) -> FieldInfo:
    """``info`` with ``changes`` to its attributes, such as its default, and constraints, such as
    ``min_length``."""
    attributes = {key: value for key, value in changes.items() if key in FieldInfo.__slots__}
    constraints = {key: value for key, value in changes.items() if key not in attributes}
    changed = FieldInfo.merge_field_infos(info, **attributes)
    return (
        FieldInfo.merge_field_infos(changed, FieldInfo(**constraints)) if constraints else changed
    )
