from typing import Literal, Self

from pydantic import Field, model_validator

from ...common.hints import hint
from ...common.text import markdown_code, plural
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import Outcome
from ..settings import Settings, limits_in_order


class LengthSettings(Settings):
    min_length: int | None = Field(
        default=None, ge=0, title="At least", description="Leave empty for no lower limit."
    )
    max_length: int | None = Field(
        default=None, ge=0, title="At most", description="Leave empty for no upper limit."
    )
    count: Literal["words", "characters"] = Field(
        default="characters",
        title="Count",
        json_schema_extra=hint(labels={"words": "Words", "characters": "Characters"}),
    )

    @model_validator(mode="after")
    def _in_order(self) -> Self:
        limits_in_order(self.min_length, self.max_length, "The highest count is below the lowest.")
        return self


class Length(AnswerMethod[LengthSettings]):
    name = "gradeflow/length"
    version = 1
    label = "Length"
    summary = "Full marks if the answer's length is within limits."
    guide = (
        "**Use it when:** The answer has a word limit, or must be long enough to count.\n\n"
        "**How it marks:** The answer's words or characters are counted, and a count from **At "
        "least** to **At most** gets full marks. Leave either empty for no limit on that side.\n\n"
        "**Related:** To check what the answer says as well, combine it with Key Points in a "
        "Combination."
    )
    question_types = frozenset({"text"})
    settings = LengthSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        low, high = settings.min_length, settings.max_length
        if low is not None and high is not None:
            return f"Has {markdown_code(low)} to {markdown_code(high)} {settings.count}."
        if low is not None:
            return f"Has at least {markdown_code(low)} {_unit(settings, low)}."
        if high is not None:
            return f"Has at most {markdown_code(high)} {_unit(settings, high)}."
        return "Any length."

    def _mark_answer(self, settings: LengthSettings, answer: Answer, question: Question) -> Outcome:
        text = str(answer)
        count = len(text.split()) if settings.count == "words" else len(text)
        said = f"{count} {_unit(settings, count)}"
        if settings.min_length is not None and count < settings.min_length:
            return Outcome(0.0, f"Too short: {said}. Expected at least {settings.min_length}.")
        if settings.max_length is not None and count > settings.max_length:
            return Outcome(0.0, f"Too long: {said}. Expected at most {settings.max_length}.")
        return Outcome(1.0, "Right length.")


def _unit(settings: LengthSettings, count: int) -> str:
    return plural(settings.count.removesuffix("s"), count)
