from typing import Literal, Self

import regex
from pydantic import Field, model_validator

from ...common.hints import hint
from ...common.problems import Problem, problem_error
from ...common.text import code_span, plain_join
from ...questions.base import Question
from ...questions.number import NumberQuestion, reads_as_number
from ...questions.reading import Reading, TextReading
from ...questions.types import Answer, Unreadable
from ...steps.example import examples_beside
from ...steps.method import TransformMethod
from ...steps.outcome import Span, Transformed, TransformResult
from ...steps.settings import Settings
from .regexes import PATTERN_TIMEOUT_MESSAGE, OneGroupPattern, pattern_capture_ranges

_NUMBER = regex.compile(
    r"(?<![\w.^])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:/\d+)?"
    r"(?:\s*[x×*]\s*10\s*(?:\^|\*\*)\s*[-+]?\d+)?%?"
)
"""A number as written among other text, not part of a name like ``Q3`` or a power like ``t^2``."""


class ExtractSettings(Settings):
    type: Literal["number", "text"] = Field(
        default="number",
        title="Type",
        json_schema_extra=hint(labels={"number": "Number", "text": "Text"}),
    )
    pattern: OneGroupPattern | None = Field(
        default=None,
        title="Pattern",
        description="A regular expression with one group around the part to find.",
        examples=[r"Answer:\s*(.+)"],
    )
    occurrence: Literal["only", "first", "last"] = Field(
        default="only",
        title="Occurrence",
        description="Which one counts when the answer has several.",
        json_schema_extra=hint(labels={"only": "Only one", "first": "First", "last": "Last"}),
    )
    nothing_found: Literal["unmarked", "no_marks"] = Field(
        default="unmarked",
        title="When nothing is found",
        json_schema_extra=hint(labels={"unmarked": "Needs marking", "no_marks": "No marks"}),
    )

    @model_validator(mode="after")
    def _check_text_has_pattern(self) -> Self:
        if self.type == "text" and self.pattern is None:
            raise problem_error(Problem("Text needs a pattern to find it by.", field="/pattern"))
        return self


class Extract(TransformMethod[ExtractSettings]):
    name = "gradeflow/extract"
    version = 1
    label = "Extract"
    summary = "Finds one part of the answer for later steps."
    guide = (
        "For an answer with several parts, such as working and a final number, when one part needs "
        "its own check.\n\n"
        "Extract gives no marks. It finds one part of the answer, and a later step in the same "
        "Workflow checks that part when you choose it as the step's **Input**.\n\n"
        "With **Type** set to **Number**, it finds numbers by itself, and a **Pattern** with one "
        "group only narrows where to look. With **Text**, it needs a pattern: a [Python regular "
        "expression](https://docs.python.org/3/library/re.html#regular-expression-syntax) with one "
        "group in brackets around the part to find. `Conclusion:\\s*(.+)` finds the text after "
        "“Conclusion:”. Patterns ignore capitals when the question's **Ignore case** is on.\n\n"
        "**Occurrence** says which one counts when there are several: **First**, **Last** or "
        "**Only one**. An answer with nothing to find needs marking, or gets no marks when **When "
        "nothing is found** is **No marks**. One with several needs marking when Occurrence is "
        "Only one.\n\n"
        "To check that a whole answer follows a format, use Pattern."
    )
    question_types = frozenset({"text"})
    settings = ExtractSettings
    examples = examples_beside(__file__)

    def describe(self, settings: ExtractSettings) -> str:
        occurrence_phrase = {"only": "the", "first": "the first", "last": "the last"}[
            settings.occurrence
        ]
        pattern_phrase = f" matching {code_span(settings.pattern)}" if settings.pattern else ""
        no_marks_phrase = ", or gives no marks" if settings.nothing_found == "no_marks" else ""
        return f"Finds {occurrence_phrase} {settings.type}{pattern_phrase}{no_marks_phrase}."

    def output_question(self, settings: ExtractSettings, question: Question) -> Question:
        """The question reading what is found in an answer to ``question``."""
        if settings.type == "text":
            return question
        return NumberQuestion(
            max_marks=question.max_marks,
            reading=Reading(blank_marker=question.reading.blank_marker),
        )

    def transform(
        self, settings: ExtractSettings, answer: Answer, question: Question
    ) -> TransformResult:
        answer_text = str(answer)
        try:
            candidates = _candidates(settings, answer_text, question)
        except TimeoutError:
            return Unreadable(PATTERN_TIMEOUT_MESSAGE)
        if not candidates or (settings.occurrence == "only" and len(candidates) > 1):
            return Unreadable(
                _extraction_failure_reason(settings, [text for text, _ in candidates]),
                marked=not candidates and settings.nothing_found == "no_marks",
            )
        text, span = candidates[-1] if settings.occurrence == "last" else candidates[0]
        if _is_blank_marker(text, question):
            return Transformed(None)
        value = self.output_question(settings, question).read(text)
        return Transformed(value, () if value is None else (span,))


def _is_blank_marker(text: str, question: Question) -> bool:
    """Whether ``text`` is the question's blank marker as its answers are read."""
    reading = question.reading
    return isinstance(reading, TextReading) and text == reading.normalized(reading.blank_marker)


def _candidates(
    settings: ExtractSettings, answer_text: str, question: Question
) -> list[tuple[str, Span]]:
    """Each part of ``answer_text``, an answer to ``question``, of the setting's type, in order,
    with its span."""
    ranges = (
        [(0, len(answer_text))]
        if settings.pattern is None
        else pattern_capture_ranges(settings.pattern, answer_text, question)
    )
    if settings.type == "text":
        reading = question.reading
        trim_spaces = isinstance(reading, TextReading) and reading.trim_spaces
        spans = [
            _trimmed(Span(start, end), answer_text) if trim_spaces else Span(start, end)
            for start, end in ranges
        ]
        return [(answer_text[span.start : span.end], span) for span in spans]
    numbers = (
        (match[0].replace(",", ""), Span(start + match.start(), start + match.end()))
        for start, end in ranges
        for match in _NUMBER.finditer(answer_text[start:end])
    )
    return [(number, span) for number, span in numbers if reads_as_number(number, question.reading)]


def _trimmed(span: Span, text: str) -> Span:
    """``span`` of ``text`` without the spaces at either end."""
    substring = text[span.start : span.end]
    start = span.start + len(substring) - len(substring.lstrip())
    return Span(start, start + len(substring.strip()))


def _extraction_failure_reason(settings: ExtractSettings, candidates: list[str]) -> str:
    if candidates:
        return f"Found several: {plain_join(candidates, conjunction='and')}."
    return "Couldn't find a number." if settings.type == "number" else "Couldn't find the text."
