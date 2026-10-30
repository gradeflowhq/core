from collections.abc import Callable
from typing import Self

from pydantic import Field, model_validator

from ...common.text import format_number, markdown_code
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import CORRECT, ROUNDING_TOLERANCE, Outcome
from ..settings import Settings, limits_in_order


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
    def _in_order(self) -> Self:
        limits_in_order(self.min_value, self.max_value, "The highest number is below the lowest.")
        return self

    def phrased(self, show: Callable[[float], str]) -> str | None:
        low, high = self.min_value, self.max_value
        if low is not None and low == high:
            return f"exactly {show(low)}"
        if low is not None and high is not None:
            return f"between {show(low)} and {show(high)}"
        if low is not None:
            return f"at least {show(low)}"
        if high is not None:
            return f"at most {show(high)}"
        return None


class NumberRange(AnswerMethod[NumberRangeSettings]):
    name = "gradeflow/number-range"
    version = 1
    label = "Number Range"
    summary = "Full marks for a number within a range."
    guide = (
        "**Use it when:** Any number in a range is right, such as an estimate, a measurement or a "
        "rounded result.\n\n"
        "**How it marks:** A number from **From** to **To**, both included, gets full marks. "
        "Leave either empty for no limit on that side.\n\n"
        "**Related:** To accept numbers near one value, use Accepted Answers with a "
        "**Tolerance**."
    )
    question_types = frozenset({"number"})
    settings = NumberRangeSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        phrase = self._settings_of(check).phrased(markdown_code)
        return f"Is {phrase}." if phrase else "Any number."

    def _mark_answer(
        self, settings: NumberRangeSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(answer, int | float)  # number questions read numbers
        below = settings.min_value is not None and answer < settings.min_value - ROUNDING_TOLERANCE
        above = settings.max_value is not None and answer > settings.max_value + ROUNDING_TOLERANCE
        if not (below or above):
            return Outcome(1.0, CORRECT)
        return Outcome(
            0.0, f"Too {'low' if below else 'high'}. Expected {settings.phrased(format_number)}."
        )
