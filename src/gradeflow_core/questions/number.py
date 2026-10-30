from typing import ClassVar, Literal

from ..common.schemas import tag
from .base import Question
from .reading import Reading, read_number, reading_field


class NumberQuestion(Question):
    label: ClassVar[str] = "Number"
    summary: ClassVar[str] = "A number, which may have a unit, such as 9.81 m/s^2."

    type: Literal["number"] = tag("number")
    reading: Reading = reading_field(Reading)

    def python_type(self) -> str:
        return "int | float | None"

    def read(self, answer: str) -> int | float | None:
        return read_number(answer, self.reading)


def reads_as_number(text: str) -> bool:
    """Whether a number question reads ``text`` as a number, not as blank or not at all."""
    try:
        return read_number(text, Reading()) is not None
    except ValueError:
        return False
