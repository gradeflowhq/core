import re

from pydantic import Field, field_validator

from ...common.hints import ADVANCED_HINT
from ...common.text import as_sentence, code_span
from ...questions.base import Question
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
    ignore_case: bool = Field(
        default=False,
        title="Ignore case",
        json_schema_extra=ADVANCED_HINT,
    )
    multi_line: bool = Field(
        default=False,
        title="Match each line",
        description="^ and $ match at each line, not only the whole answer.",
        json_schema_extra=ADVANCED_HINT,
    )
    dotall: bool = Field(
        default=False,
        title="Dot matches line breaks",
        json_schema_extra=ADVANCED_HINT,
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
        "pattern with `^` and end it with `$` to match the whole answer.\n\n"
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
        settings = self._settings_of(check)
        flag_notes = [
            text
            for enabled, text in (
                (settings.ignore_case, "ignoring case"),
                (settings.multi_line, "matching each line"),
                (settings.dotall, "with a dot matching line breaks"),
            )
            if enabled
        ]
        return as_sentence(
            ", ".join([f"Matches the pattern {code_span(settings.pattern)}", *flag_notes])
        )

    def _mark_answer(
        self, settings: PatternSettings, answer: Answer, question: Question
    ) -> Outcome:
        flags = (
            (re.IGNORECASE if settings.ignore_case else re.NOFLAG)
            | (re.MULTILINE if settings.multi_line else re.NOFLAG)
            | (re.DOTALL if settings.dotall else re.NOFLAG)
        )
        is_match = re.search(settings.pattern, str(answer), flags) is not None
        return Outcome(
            float(is_match), "Correct format." if is_match else "Not in the expected format."
        )
