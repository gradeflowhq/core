"""Custom Code's parameters: named values its code reads as variables, each with its type."""

import keyword
import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AfterValidator, ConfigDict, Field, JsonValue

from ...common.hints import hint
from ...common.schemas import type_field
from ..settings import IDENTIFIER, Settings

RESERVED_NAMES = frozenset({"answer", "answers", "results", "correct", "proportion", "feedback"})


class IntegerParameter(Settings):
    model_config = ConfigDict(title="Whole number")
    type: Literal["integer"] = type_field()
    value: int = Field(title="Value", description="A whole number, such as 3.")


class NumberParameter(Settings):
    model_config = ConfigDict(title="Number")
    type: Literal["number"] = type_field()
    value: float = Field(title="Value", description="A number, such as 2.5.")


class TextParameter(Settings):
    model_config = ConfigDict(title="Text")
    type: Literal["text"] = type_field()
    value: str = Field(title="Value", description="Any text.")


class BooleanParameter(Settings):
    model_config = ConfigDict(title="Yes/No")
    type: Literal["boolean"] = type_field()
    value: bool = Field(title="Value", description="Yes or no.")


class ListParameter(Settings):
    model_config = ConfigDict(title="List")
    type: Literal["list"] = type_field()
    value: list[Parameter] = Field(title="Items", description="Values of one type.")


class TableParameter(Settings):
    model_config = ConfigDict(title="Table")
    type: Literal["table"] = type_field()
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


def _validated_parameter_name(name: str) -> str:
    if not re.fullmatch(IDENTIFIER, name):
        raise ValueError("Use letters, digits and underscores, not starting with a digit.")
    if keyword.iskeyword(name):
        raise ValueError(f"{name} is a Python keyword; choose another name.")
    if name in RESERVED_NAMES:
        raise ValueError(f"Your code already uses {name}; choose another name.")
    return name


ParameterName = Annotated[str, AfterValidator(_validated_parameter_name)]

PARAMETERS_FIELD: dict[ParameterName, Parameter] = Field(
    default_factory=dict,
    title="Parameters",
    description="Named values your code can use as variables.",
    json_schema_extra={
        "propertyNames": {"pattern": IDENTIFIER},
        **hint(item_label="parameter"),
    },
)


def parameter_values(parameters: Mapping[str, Parameter]) -> dict[str, JsonValue]:
    """The parameters as the plain values the code sees."""
    return {name: _plain_value(parameter) for name, parameter in parameters.items()}


def _plain_value(parameter: Parameter) -> JsonValue:
    if isinstance(parameter, ListParameter):
        return [_plain_value(nested_parameter) for nested_parameter in parameter.value]
    if isinstance(parameter, TableParameter):
        return {
            key: _plain_value(nested_parameter) for key, nested_parameter in parameter.value.items()
        }
    return parameter.value
