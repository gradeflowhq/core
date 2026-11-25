from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import Field

from ..common.hints import hint
from ..common.schemas import JsonDict, type_field
from .base import Question
from .number import NumberQuestion
from .parts import (
    MIN_PARTS,
    AnswerParts,
    part_properties,
    part_types,
    read_parts,
    unbracketed,
    value_question,
)
from .reading import ListReading, read_list, reading_field
from .text import TextQuestion
from .types import PartValues


class MultipartQuestion(Question):
    label: ClassVar[str] = "Multi-part"
    summary: ClassVar[str] = "An answer in named parts, such as a place and a year."
    new_question_data: ClassVar[dict[str, Any]] = {
        "parts": [{"name": "part_1"}, {"name": "part_2"}]
    }

    type: Literal["multipart"] = type_field("multipart")
    reading: ListReading = reading_field(ListReading)
    parts: AnswerParts = Field(
        min_length=MIN_PARTS,
        title="Parts",
        description="What an answer is made of, in the order it is written.",
        json_schema_extra=hint(item_label="part"),
    )

    def part_question(self, name: str) -> TextQuestion | NumberQuestion | None:
        """A question reading part ``name`` of answers as this one reads it; None if there is no
        such part."""
        return next(
            (
                value_question(part.type, self.reading, self.max_marks)
                for part in self.parts
                if part.name == name
            ),
            None,
        )

    def python_type(self) -> str:
        return f"dict[str, str | int | float | None] | None, with {part_types(self.parts)}"

    def read(self, answer: str) -> PartValues | None:
        values = read_list(unbracketed(answer), self.reading)
        if values is None:
            return None
        part_values = read_parts(self.parts, self._one_per_part(values), self.reading)
        return part_values if any(value is not None for value in part_values.values()) else None

    def answer_schema(self) -> JsonDict:
        return {"type": "object", "properties": part_properties(self.parts)}

    def answer_values(self, answer: str) -> dict[str, Any]:
        written_values = (
            self._one_per_part(unbracketed(answer).split(self.reading.separator))
            if answer.strip()
            else [""] * len(self.parts)
        )
        return {
            part.name: value.strip() for part, value in zip(self.parts, written_values, strict=True)
        }

    def answer_from(self, values: Mapping[str, Any]) -> str:
        written_values = [str(values.get(part.name) or "").strip() for part in self.parts]
        return self.reading.separator.join(written_values) if any(written_values) else ""

    def _one_per_part(self, values: list[str]) -> list[str]:
        if len(values) != len(self.parts):
            raise ValueError(f"Expected {len(self.parts)} parts, but got {len(values)}.")
        return values
