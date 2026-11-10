from typing import ClassVar, Literal

from ..common.schemas import type_field
from .base import Question
from .reading import TextReading, is_blank, reading_field


class TextQuestion(Question):
    label: ClassVar[str] = "Text"
    summary: ClassVar[str] = "Any written answer, such as a word or a sentence."

    type: Literal["text"] = type_field("text")
    reading: TextReading = reading_field(TextReading)

    def python_type(self) -> str:
        return "str | None"

    def read(self, answer: str) -> str | None:
        # Case is changed after the blank check, so a blank marker in capitals still counts.
        if self.reading.trim_spaces:
            answer = answer.strip()
        return None if is_blank(answer, self.reading) else self.reading.normalized(answer)
