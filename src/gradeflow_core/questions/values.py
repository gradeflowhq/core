from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import Field

from ..common.hints import hint
from ..common.python_layouts import BRACKETS
from ..common.schemas import JsonDict, type_field
from .base import Question
from .number import NumberQuestion
from .reading import ListReading, Reading, TextReading, read_list, reading_field
from .text import TextQuestion
from .types import Value, ValueType

VALUE_TYPE_LABELS: dict[str, str] = {"text": "Text", "number": "Number"}
EXAMSOFT_SEPARATOR = "~"
"""ExamSoft's separator between an answer's values."""


class ValuesQuestion(Question):
    label: ClassVar[str] = "Multiple values"
    summary: ClassVar[str] = "Several values in one answer, such as 3, -2."
    new_question_data: ClassVar[dict[str, Any]] = {"value_types": ["text", "text"]}

    type: Literal["values"] = type_field("values")
    reading: ListReading = reading_field(ListReading)
    value_types: list[ValueType] = Field(
        min_length=1,
        title="Values",
        description="What each value in an answer is, in order.",
        json_schema_extra=hint(labels=VALUE_TYPE_LABELS, item_label="value"),
    )

    @property
    def value_count(self) -> int:
        return len(self.value_types)

    def value_question(self, number: int) -> TextQuestion | NumberQuestion | None:
        if not 1 <= number <= self.value_count:
            return None
        return self._value_questions()[number - 1]

    def _value_questions(self) -> list[TextQuestion | NumberQuestion]:
        number_question = NumberQuestion(
            max_marks=self.max_marks, reading=Reading(blank_marker=self.reading.blank_marker)
        )
        text_question = TextQuestion(
            max_marks=self.max_marks,
            reading=TextReading(**self.reading.model_dump(exclude={"separator"})),
        )
        return [
            number_question if value_type == "number" else text_question
            for value_type in self.value_types
        ]

    def answer_schema(self) -> JsonDict:
        properties = {
            f"value_{number}": {
                "type": "string",
                "title": f"Value {number}, {VALUE_TYPE_LABELS[value_type].lower()}",
            }
            for number, value_type in enumerate(self.value_types, 1)
        }
        return {"type": "object", "properties": properties}

    def answer_values(self, answer: str) -> dict[str, Any]:
        written_values = (
            self._counted(_unbracketed(answer).split(self.reading.separator))
            if answer.strip()
            else [""] * self.value_count
        )
        return {
            f"value_{number}": value.strip()
            for number, value in zip(self._value_numbers, written_values, strict=True)
        }

    def answer_from(self, values: Mapping[str, Any]) -> str:
        written_values = [
            str(values.get(f"value_{number}") or "").strip() for number in self._value_numbers
        ]
        return self.reading.separator.join(written_values) if any(written_values) else ""

    @property
    def _value_numbers(self) -> range:
        return range(1, self.value_count + 1)

    def _counted(self, values: list[str]) -> list[str]:
        if len(values) != self.value_count:
            raise ValueError(f"Expected {self.value_count} values, but got {len(values)}.")
        return values

    def python_type(self) -> str:
        return "list[str | int | float | None], in order: " + ", ".join(self.value_types)

    def read(self, answer: str) -> list[Value] | None:
        values = read_list(_unbracketed(answer), self.reading)
        if values is None:
            return None
        return [
            question.read(value)
            for question, value in zip(self._value_questions(), self._counted(values), strict=True)
        ]


def _unbracketed(answer: str) -> str:
    """The values inside one pair of brackets around them, as in `(3, -2)`."""
    text = answer.strip()
    if len(text) > 1 and BRACKETS.get(text[0]) == text[-1]:
        return text[1:-1]
    return answer
