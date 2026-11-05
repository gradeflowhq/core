import re

from pydantic import Field, field_validator

from ...common.text import code_span
from ...questions.base import Question
from ...questions.text import TextQuestion
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import Outcome
from ..settings import Settings


class PatternSettings(Settings):
    pattern: str = Field(
        min_length=1,
        title="Pattern",
        examples=["^colou?r$"],
    )

    @field_validator("pattern")
    @classmethod
    def _check_compiles(cls, pattern: str) -> str:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"Not a valid pattern: {error}.") from None
        return pattern


class Pattern(AnswerMethod[PatternSettings]):
    name = "gradeflow/pattern"
    version = 1
    label = "Pattern"
    summary = "Full marks if the answer matches a regular expression."
    guide = (
        "**Use it when:** Correct answers follow a pattern that a list can't cover, such as any "
        "date written as DD/MM/YYYY.\n\n"
        "**How it marks:** The answer gets full marks if the pattern is found anywhere in it. "
        "Patterns use [Python's regular expression syntax]"
        "(https://docs.python.org/3/library/re.html#regular-expression-syntax). Start the "
        "pattern with `^` and end it with `$` to match the whole answer. On a question that keeps "
        "case, start it with `(?i)` to ignore case.\n\n"
        "**Good to know:** A pattern checks the form, not the value, so `6/8` matches "
        "`^\\d+/\\d+$`. Pair it with a check of the value in a Combination.\n\n"
        "**Related:** When a list of answers or words will do, use Accepted Answers or Key "
        "Points: they're easier to read later."
    )
    level = "advanced"
    question_types = frozenset({"text", "code"})
    settings = PatternSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        return f"Matches the pattern {code_span(self._settings_of(check).pattern)}."

    def _mark_answer(
        self, settings: PatternSettings, answer: Answer, question: Question
    ) -> Outcome:
        # Answers to a question that ignores case are lowercased, so the pattern ignores it too.
        ignores_case = isinstance(question, TextQuestion) and question.reading.ignore_case
        flags = re.IGNORECASE if ignores_case else re.NOFLAG
        is_match = re.search(settings.pattern, str(answer), flags) is not None
        return Outcome(
            float(is_match), "Correct format." if is_match else "Not in the expected format."
        )
