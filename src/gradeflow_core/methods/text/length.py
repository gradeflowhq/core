from typing import Literal, Self

from pydantic import Field, model_validator

from ...common.hints import hint
from ...common.text import code_span, plural
from ...questions.base import Question
from ...questions.types import Answer
from ...steps.example import examples_beside
from ...steps.method import AnswerMethod
from ...steps.outcome import Outcome
from ...steps.settings import Settings, check_limits_in_order


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
            self.min_length,
            self.max_length,
            "The highest count is below the lowest.",
            "/max_length",
        )
        return self


class Length(AnswerMethod[LengthSettings]):
    name = "gradeflow/length"
    version = 1
    label = "Length"
    summary = "Correct if the answer's length is within limits."
    guide = (
        "For a word limit, or a minimum length an answer needs to count.\n\n"
        "**Count** chooses words or characters, and counts characters unless you set it to "
        "**Words**. A count from **At least** to **At most** is correct, and both ends count. "
        "Leave either empty for no limit on that side.\n\n"
        "To check what the answer says as well, put Length and Key Points in a Workflow with **All "
        "must be correct**."
    )
    question_types = frozenset({"text"})
    settings = LengthSettings
    examples = examples_beside(__file__)

    def describe(self, settings: LengthSettings) -> str:
        min_length, max_length = settings.min_length, settings.max_length
        if min_length is not None and max_length is not None:
            return f"Has {code_span(min_length)} to {code_span(max_length)} {settings.unit}."
        if min_length is not None:
            return f"Has at least {code_span(min_length)} {_unit_name(settings, min_length)}."
        if max_length is not None:
            return f"Has at most {code_span(max_length)} {_unit_name(settings, max_length)}."
        return "Any length."

    def mark_answer(self, settings: LengthSettings, answer: Answer, question: Question) -> Outcome:
        answer_text = str(answer)
        length = len(answer_text.split()) if settings.unit == "words" else len(answer_text)
        length_text = f"{length} {_unit_name(settings, length)}"
        if settings.min_length is not None and length < settings.min_length:
            return Outcome(
                0.0, f"Too short: {length_text}. Expected at least {settings.min_length}."
            )
        if settings.max_length is not None and length > settings.max_length:
            return Outcome(0.0, f"Too long: {length_text}. Expected at most {settings.max_length}.")
        return Outcome(1.0, "Correct length.")


def _unit_name(settings: LengthSettings, count: int) -> str:
    return plural(settings.unit.removesuffix("s"), count)
