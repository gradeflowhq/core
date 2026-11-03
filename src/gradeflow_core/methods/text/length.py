from typing import Literal, Self

from pydantic import Field, model_validator

from ...common.hints import hint
from ...common.text import code_span, plural
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import Outcome
from ..settings import Settings, check_limits_in_order


class LengthSettings(Settings):
    min_length: int | None = Field(
        default=None, ge=0, title="At least", description="Leave empty for no lower limit."
    )
    max_length: int | None = Field(
        default=None, ge=0, title="At most", description="Leave empty for no upper limit."
    )
    unit: Literal["words", "characters"] = Field(
        default="characters",
        title="Count",
        json_schema_extra=hint(labels={"words": "Words", "characters": "Characters"}),
    )

    @model_validator(mode="after")
    def _check_limits_in_order(self) -> Self:
        check_limits_in_order(
            self.min_length, self.max_length, "The highest count is below the lowest."
        )
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
        min_length, max_length = settings.min_length, settings.max_length
        if min_length is not None and max_length is not None:
            return f"Has {code_span(min_length)} to {code_span(max_length)} {settings.unit}."
        if min_length is not None:
            return f"Has at least {code_span(min_length)} {_unit_name(settings, min_length)}."
        if max_length is not None:
            return f"Has at most {code_span(max_length)} {_unit_name(settings, max_length)}."
        return "Any length."

    def _mark_answer(self, settings: LengthSettings, answer: Answer, question: Question) -> Outcome:
        text = str(answer)
        count = len(text.split()) if settings.unit == "words" else len(text)
        count_text = f"{count} {_unit_name(settings, count)}"
        if settings.min_length is not None and count < settings.min_length:
            return Outcome(
                0.0, f"Too short: {count_text}. Expected at least {settings.min_length}."
            )
        if settings.max_length is not None and count > settings.max_length:
            return Outcome(0.0, f"Too long: {count_text}. Expected at most {settings.max_length}.")
        return Outcome(1.0, "Correct length.")


def _unit_name(settings: LengthSettings, count: int) -> str:
    return plural(settings.unit.removesuffix("s"), count)
