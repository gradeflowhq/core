from typing import ClassVar, Literal, Self

from pydantic import Field

from ..common.hints import hint
from ..common.schemas import JsonDict, tag
from .base import Question
from .reading import CodeReading, read_code, reading_field
from .types import Answer, AnswerShape


class CodeQuestion(Question):
    label: ClassVar[str] = "Code"
    summary: ClassVar[str] = "A program, such as a Python function."

    type: Literal["code"] = tag("code")
    language: Literal["python"] = Field(
        default="python", title="Language", description="What the answers are written in."
    )
    reading: CodeReading = reading_field(CodeReading)

    @property
    def answer_shape(self) -> AnswerShape:
        return "multiple" if self.reading.guess_indentation else "single"

    def single(self) -> Self:
        return self.model_copy(
            update={"reading": self.reading.model_copy(update={"guess_indentation": False})}
        )

    def python_type(self) -> str:
        if self.answer_shape == "multiple":
            return "list[str] | None, the likeliest layout first"
        return "str | None"

    def read(self, answer: str) -> str | list[str] | None:
        return read_code(answer, self.reading)

    def first(self, answer: Answer) -> Answer:
        return answer[0] if isinstance(answer, list) and answer else answer

    def answer_schema(self) -> JsonDict:
        return {"type": "string", **hint(kind="code", language=self.language)}
