import math
from collections.abc import Callable
from typing import Self

from pydantic import Field, model_validator

from ...common.text import code_span, format_number
from ...questions.base import Question
from ...questions.types import Answer
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import CORRECT_FEEDBACK, Outcome
from ..settings import Settings, check_limits_in_order
from ..step import Step


class NumberRangeSettings(Settings):
    min_value: float | None = Field(
        default=None,
        title="From",
        description="Leave empty for no lower limit.",
    )
    max_value: float | None = Field(
        default=None,
        title="To",
        description="Leave empty for no upper limit.",
    )

    @model_validator(mode="after")
    def _check_limits_in_order(self) -> Self:
        check_limits_in_order(
            self.min_value, self.max_value, "The highest number is below the lowest."
        )
        return self

    def range_phrase(self, format_value: Callable[[float], str]) -> str | None:
        min_value, max_value = self.min_value, self.max_value
        if min_value is not None and min_value == max_value:
            return f"exactly {format_value(min_value)}"
        if min_value is not None and max_value is not None:
            return f"between {format_value(min_value)} and {format_value(max_value)}"
        if min_value is not None:
            return f"at least {format_value(min_value)}"
        if max_value is not None:
            return f"at most {format_value(max_value)}"
        return None


class NumberRange(AnswerMethod[NumberRangeSettings]):
    name = "gradeflow/number-range"
    version = 1
    label = "Number Range"
    summary = "Full marks for a number within a range."
    guide = (
        "**Use it when:** Any number in a range is correct, such as an estimate, a measurement or "
        "a rounded result.\n\n"
        "**How it marks:** A number from **From** to **To**, both included, gets full marks. "
        "Leave either empty for no limit on that side.\n\n"
        "**Related:** To accept numbers near one value, use Accepted Answers with a "
        "**Tolerance**."
    )
    question_types = frozenset({"number"})
    settings = NumberRangeSettings
    examples = examples_beside(__file__)

    def describe(self, step: Step) -> str:
        phrase = self._settings_of(step).range_phrase(code_span)
        return f"Is {phrase}." if phrase else "Any number."

    def _mark_answer(
        self, settings: NumberRangeSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(answer, int | float)  # number questions read numbers
        min_value, max_value = settings.min_value, settings.max_value
        too_low = (
            min_value is not None and answer < min_value and not math.isclose(answer, min_value)
        )
        too_high = (
            max_value is not None and answer > max_value and not math.isclose(answer, max_value)
        )
        if not (too_low or too_high):
            return Outcome(1.0, CORRECT_FEEDBACK)
        expected_range = settings.range_phrase(format_number)
        return Outcome(0.0, f"Too {'low' if too_low else 'high'}. Expected {expected_range}.")
