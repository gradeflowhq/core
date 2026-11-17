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
    summary = "Correct if the answer matches a regular expression."
    guide = (
        "For answers that must follow a format a list can't cover, such as a date written as "
        "DD/MM/YYYY.\n\n"
        "The answer is correct if the **Pattern** is found anywhere in it. Patterns are [Python "
        "regular "
        "expressions](https://docs.python.org/3/library/re.html#regular-expression-syntax). Start "
        "with `^` and end with `$` to match the whole answer: `^\\d{2}/\\d{2}/\\d{4}$` accepts "
        "`07/03/2025` but not `On 07/03/2025`.\n\n"
        "A pattern checks the form, not the value: `^\\d+/\\d+$` accepts `6/8` and `9/8` alike. To "
        "check the value as well, add another check beside it in a Workflow. Patterns ignore "
        "capitals when the question's **Ignore case** is on. Otherwise, start the pattern with "
        "`(?i)` to ignore them.\n\n"
        "When a list of answers or words will do, use Accepted Answers or Key Points instead."
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
