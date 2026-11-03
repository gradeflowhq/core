from typing import ClassVar, Literal

from pydantic import Field

from ..common.hints import hint
from ..common.schemas import JsonDict, tag
from .base import Question
from .reading import Reading, read_code, reading_field


class CodeQuestion(Question):
    label: ClassVar[str] = "Code"
    summary: ClassVar[str] = "A program, such as a Python function."

    type: Literal["code"] = tag("code")
    language: Literal["python"] = Field(
        default="python", title="Language", description="What the answers are written in."
    )
    reading: Reading = reading_field(Reading)

    def python_type(self) -> str:
        return "str | None"

    def read(self, answer: str) -> str | None:
        return read_code(answer, self.reading)

    def answer_schema(self) -> JsonDict:
        return {"type": "string", **hint(kind="code", language=self.language)}
