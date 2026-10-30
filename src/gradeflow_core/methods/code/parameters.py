"""Custom Code's parameters: named values its code reads as variables, each with its type."""

import keyword
import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AfterValidator, ConfigDict, Field, JsonValue

from ...common.hints import hint
from ...common.schemas import tag
from ..settings import IDENTIFIER, Settings

RESERVED = frozenset({"answer", "answers", "results", "correct", "proportion", "feedback"})


class IntegerParameter(Settings):
    model_config = ConfigDict(title="Whole number")
    type: Literal["integer"] = tag()
    value: int = Field(title="Value", description="A whole number, such as 3.")


class NumberParameter(Settings):
    model_config = ConfigDict(title="Number")
    type: Literal["number"] = tag()
    value: float = Field(title="Value", description="A number, such as 2.5.")


class TextParameter(Settings):
    model_config = ConfigDict(title="Text")
    type: Literal["text"] = tag()
    value: str = Field(title="Value", description="Any text.")


class BooleanParameter(Settings):
    model_config = ConfigDict(title="Yes/No")
    type: Literal["boolean"] = tag()
    value: bool = Field(title="Value", description="Yes or no.")


class ListParameter(Settings):
    model_config = ConfigDict(title="List")
    type: Literal["list"] = tag()
    value: list[Parameter] = Field(title="Items", description="Values of one type.")


class TableParameter(Settings):
    model_config = ConfigDict(title="Table")
    type: Literal["table"] = tag()
    value: dict[str, Parameter] = Field(title="Rows", description="Rows of name and value.")


Parameter = Annotated[
    IntegerParameter
    | NumberParameter
    | TextParameter
    | BooleanParameter
    | ListParameter
    | TableParameter,
    Field(discriminator="type"),
]
ListParameter.model_rebuild()
TableParameter.model_rebuild()


def _named(name: str) -> str:
    if not re.fullmatch(IDENTIFIER, name):
        raise ValueError("Use letters, digits and underscores, not starting with a digit.")
    if keyword.iskeyword(name):
        raise ValueError(f"{name} is a Python keyword; choose another name.")
    if name in RESERVED:
        raise ValueError(f"Your code already uses {name}; choose another name.")
    return name


ParameterName = Annotated[str, AfterValidator(_named)]

PARAMETERS: dict[ParameterName, Parameter] = Field(
    default_factory=dict,
    title="Parameters",
    description="Named values your code can use as variables.",
    json_schema_extra={
        "propertyNames": {"pattern": IDENTIFIER},
        **hint(item="parameter"),
    },
)


def values(parameters: Mapping[str, Parameter]) -> dict[str, JsonValue]:
    """The parameters as the plain values the code sees."""
    return {name: _value(parameter) for name, parameter in parameters.items()}


def _value(parameter: Parameter) -> JsonValue:
    if isinstance(parameter, ListParameter):
        return [_value(item) for item in parameter.value]
    if isinstance(parameter, TableParameter):
        return {key: _value(item) for key, item in parameter.value.items()}
    return parameter.value
