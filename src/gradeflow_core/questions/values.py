from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import Field

from ..common.hints import hint
from ..common.schemas import JsonDict, tag
from .base import Question
from .number import NumberQuestion
from .reading import ListReading, Reading, TextReading, read_list, reading_field
from .text import TextQuestion
from .types import Value, ValueType

VALUE_TYPE_LABELS: dict[ValueType, str] = {"text": "Text", "number": "Number"}
BLANK_SEPARATOR = "~"
"""What separates the blanks of an ExamSoft answer key, and the values of answers to them."""


def values_reading() -> ListReading:
    """How values are read when an answer key sets the question up: separated by ``~``."""
    return ListReading(separator=BLANK_SEPARATOR)


class ValuesQuestion(Question):
    label: ClassVar[str] = "Multiple values"
    summary: ClassVar[str] = "Several values in one answer, such as 3, -2."
    starting: ClassVar[dict[str, Any]] = {"value_types": ["text", "text"]}

    type: Literal["values"] = tag("values")
    reading: ListReading = reading_field(ListReading)
    value_types: list[ValueType] = Field(
        ...,
        min_length=1,
        title="Values",
        description="What each value in an answer is, in order.",
        json_schema_extra=hint(labels=VALUE_TYPE_LABELS, item="value"),
    )

    @property
    def value_count(self) -> int:
        return len(self.value_types)

    def value_question(self, value: int) -> TextQuestion | NumberQuestion | None:
        if not 1 <= value <= self.value_count:
            return None
        return self._value_questions()[value - 1]

    def _value_questions(self) -> list[TextQuestion | NumberQuestion]:
        number = NumberQuestion(
            max_marks=self.max_marks, reading=Reading(blank_marker=self.reading.blank_marker)
        )
        text = TextQuestion(
            max_marks=self.max_marks,
            reading=TextReading(**self.reading.model_dump(exclude={"separator"})),
        )
        return [number if kind == "number" else text for kind in self.value_types]

    def answer_schema(self) -> JsonDict:
        properties = {
            f"value_{number}": {
                "type": "string",
                "title": f"Value {number}, {VALUE_TYPE_LABELS[kind].lower()}",
            }
            for number, kind in enumerate(self.value_types, 1)
        }
        return {"type": "object", "properties": properties}

    def answer_values(self, answer: str) -> dict[str, Any]:
        written = _unbracketed(answer).split(self.reading.separator) if answer.strip() else []
        written += [""] * (self.value_count - len(written))
        return {f"value_{number}": written[number - 1].strip() for number in self._numbers}

    def answer_from(self, values: Mapping[str, Any]) -> str:
        written = [str(values.get(f"value_{number}") or "").strip() for number in self._numbers]
        return self.reading.separator.join(written) if any(written) else ""

    @property
    def _numbers(self) -> range:
        return range(1, self.value_count + 1)

    def python_type(self) -> str:
        return "list[str | int | float | None], in order: " + ", ".join(self.value_types)

    def read(self, answer: str) -> list[Value] | None:
        values = read_list(_unbracketed(answer), self.reading)
        if values is None:
            return None
        if len(values) != self.value_count:
            raise ValueError(f"Expected {self.value_count} values, but got {len(values)}.")
        return [
            question.read(value)
            for question, value in zip(self._value_questions(), values, strict=True)
        ]


_BRACKETS = {"(": ")", "[": "]", "{": "}"}


def _unbracketed(raw: str) -> str:
    """The values inside one pair of brackets around them, as in `(3, -2)`."""
    text = raw.strip()
    if len(text) > 1 and _BRACKETS.get(text[0]) == text[-1]:
        return text[1:-1]
    return raw
