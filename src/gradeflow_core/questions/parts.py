"""The named parts of multi-part answers and of list items, and how their values are read."""

from collections.abc import Sequence
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from ..common.error_formatting import IDENTIFIER
from ..common.hints import hint
from ..common.python_layouts import BRACKETS
from ..common.schemas import JsonDict
from ..common.text import capitalized
from .number import NumberQuestion
from .reading import ListReading, Reading, TextReading
from .text import TextQuestion
from .types import PartValues, ValueType

VALUE_TYPE_LABELS: dict[str, str] = {"text": "Text", "number": "Number"}
MIN_PARTS = 2


class AnswerPart(BaseModel):
    model_config = ConfigDict(extra="forbid", title="Part")

    name: str = Field(
        pattern=IDENTIFIER,
        title="Name",
        description="A check reads it as `{{ answer.year }}`.",
    )
    type: ValueType = Field(
        default="text", title="Type", json_schema_extra=hint(labels=VALUE_TYPE_LABELS)
    )


def part_label(name: str) -> str:
    """How feedback and the editor show the part named ``name``: ``atomic_number`` as "Atomic
    number"."""
    return capitalized(name.replace("_", " "))


def _with_unique_names(parts: list[AnswerPart]) -> list[AnswerPart]:
    names = [part.name for part in parts]
    if len(set(names)) < len(names):
        raise ValueError("Give each part its own name.")
    return parts


AnswerParts = Annotated[list[AnswerPart], AfterValidator(_with_unique_names)]


def part_types(parts: Sequence[AnswerPart]) -> str:
    """Each part's name and type, as custom code's answer type lists them: ``place: text, year:
    number``."""
    return ", ".join(f"{part.name}: {part.type}" for part in parts)


def part_properties(parts: Sequence[AnswerPart]) -> JsonDict:
    """The answer schema's property of each part, titled by its label."""
    return {part.name: {"type": "string", "title": part_label(part.name)} for part in parts}


def numbered_parts(value_types: Sequence[ValueType]) -> list[AnswerPart]:
    """A part of each type, named by its place: ``part_1``, ``part_2``."""
    return [
        AnswerPart(name=f"part_{number}", type=value_type)
        for number, value_type in enumerate(value_types, 1)
    ]


def value_question(
    value_type: ValueType, reading: ListReading, max_marks: float
) -> TextQuestion | NumberQuestion:
    """A question reading one value of an answer as ``reading`` reads its values."""
    if value_type == "number":
        return NumberQuestion(
            max_marks=max_marks, reading=Reading(blank_marker=reading.blank_marker)
        )
    return TextQuestion(
        max_marks=max_marks, reading=TextReading(**reading.model_dump(exclude={"separator"}))
    )


def read_parts(
    parts: Sequence[AnswerPart], values: Sequence[str], reading: ListReading
) -> PartValues:
    """Each of ``values``, written in the order of ``parts``, read as its part's type."""
    return {
        part.name: value_question(part.type, reading, 1.0).read(value)
        for part, value in zip(parts, values, strict=True)
    }


def unbracketed(answer: str) -> str:
    """The values inside one pair of brackets around them, as in `(3, -2)`."""
    text = answer.strip()
    if len(text) > 1 and BRACKETS.get(text[0]) == text[-1]:
        return text[1:-1]
    return answer
