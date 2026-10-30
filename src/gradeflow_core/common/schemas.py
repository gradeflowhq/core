"""JSON schemas of what the core accepts."""

from typing import Any

from pydantic import Field, TypeAdapter
from pydantic_core import PydanticUndefined

type JsonDict = dict[str, Any]


def tag(default: Any = PydanticUndefined) -> Any:
    """A fixed field that says what kind of object holds it, such as a question's type. Without a
    default, an object written without its defaults still says it."""
    return Field(default=default, frozen=True, json_schema_extra={"readOnly": True})


def json_schema(model: Any) -> JsonDict:  # Any: what TypeAdapter takes
    """The JSON schema of what ``model`` accepts, with a union of plain types as a type array."""
    return TypeAdapter(model).json_schema(mode="validation", union_format="primitive_type_array")
