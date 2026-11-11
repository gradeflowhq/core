from pydantic import Field

from ...common.text import code_span
from ...questions.base import Question
from ...questions.types import Answer
from ...steps.example import examples_beside
from ...steps.method import AnswerMethod
from ...steps.outcome import Outcome
from ...steps.settings import Settings
from .regexes import PATTERN_TIMEOUT_MESSAGE, ValidPattern, pattern_matches


class PatternSettings(Settings):
    pattern: ValidPattern = Field(
        min_length=1,
        title="Pattern",
        examples=["^colou?r$"],
    )


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
        "`^\\d+/\\d+$`. Pair it with a check of the value in a Workflow.\n\n"
        "**Related:** When a list of answers or words will do, use Accepted Answers or Key "
        "Points: they're easier to read later."
    )
    level = "advanced"
    question_types = frozenset({"text", "code"})
    settings = PatternSettings
    examples = examples_beside(__file__)

    def describe(self, settings: PatternSettings) -> str:
        return f"Matches the pattern {code_span(settings.pattern)}."

    def mark_answer(self, settings: PatternSettings, answer: Answer, question: Question) -> Outcome:
        try:
            is_match = pattern_matches(settings.pattern, str(answer), question)
        except TimeoutError:
            return Outcome(0.0, PATTERN_TIMEOUT_MESSAGE, marked=False)
        return Outcome(
            float(is_match), "Correct format." if is_match else "Not in the expected format."
        )
