from typing import ClassVar, Literal

from ..common.schemas import tag
from .base import Question
from .reading import TextReading, read_text, reading_field


class TextQuestion(Question):
    label: ClassVar[str] = "Text"
    summary: ClassVar[str] = "Any written answer, such as a word or a sentence."

    type: Literal["text"] = tag("text")
    reading: TextReading = reading_field(TextReading)

    def python_type(self) -> str:
        return "str | None"

    def read(self, answer: str) -> str | None:
        return read_text(answer, self.reading)
