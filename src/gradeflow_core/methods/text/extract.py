from collections.abc import Mapping
from typing import Any, Literal, Self

import regex
from pydantic import Field, field_validator, model_validator

from ...common.hints import hint
from ...common.text import code_span, plain_join
from ...questions.base import Question
from ...questions.number import NumberQuestion, reads_as_number
from ...questions.reading import Reading, TextReading
from ...questions.types import Answer, Unreadable
from ..example import examples_beside
from ..method import Method, TransformMethod
from ..outcome import TransformResult
from ..settings import Settings
from ..step import Step
from .regexes import PATTERN_TIMEOUT_MESSAGE, compile_pattern, pattern_captures

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
    pattern: str | None = Field(
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

    @field_validator("pattern")
    @classmethod
    def _validate_one_group(cls, pattern: str | None) -> str | None:
        if pattern is not None and compile_pattern(pattern).groups != 1:
            raise ValueError("Write a pattern with one group, in brackets.")
        return pattern

    @model_validator(mode="after")
    def _validate_text_has_pattern(self) -> Self:
        if self.type == "text" and self.pattern is None:
            raise ValueError("Text needs a pattern to find it by.")
        return self


class Extract(TransformMethod[ExtractSettings]):
    name = "gradeflow/extract"
    version = 1
    label = "Extract"
    summary = "Finds one part of the answer for later steps."
    guide = (
        "**Use it when:** An answer holds several parts, such as reasoning and a final number, "
        "and one part needs its own check.\n\n"
        "**How it marks:** It gives no marks. It finds a number or a piece of text, and a later "
        "step reads it as its **Input**. An answer with nothing to find, or several where "
        "**Occurrence** says only one, is left for you to mark.\n\n"
        "**Good to know:** Numbers are found without a **Pattern**. A pattern narrows where to "
        "look, and uses [Python's regular expression syntax]"
        "(https://docs.python.org/3/library/re.html#regular-expression-syntax) with one group "
        "around the part to find.\n\n"
        "**Related:** To check the form of a whole answer, use Pattern."
    )
    question_types = frozenset({"text"})
    settings = ExtractSettings
    examples = examples_beside(__file__)

    def describe(self, step: Step) -> str:
        settings = self._settings_of(step)
        occurrence_phrase = {"only": "the", "first": "the first", "last": "the last"}[
            settings.occurrence
        ]
        pattern_phrase = f" matching {code_span(settings.pattern)}" if settings.pattern else ""
        return f"Finds {occurrence_phrase} {settings.type}{pattern_phrase}."

    def output_question(
        self,
        settings: Mapping[str, Any],
        source_question: Question,
        methods: Mapping[str, Method[Any]],
    ) -> Question:
        if settings.get("type", "number") == "text":
            return source_question
        return NumberQuestion(
            max_marks=source_question.max_marks,
            reading=Reading(blank_marker=source_question.reading.blank_marker),
        )

    def _transform(
        self, settings: ExtractSettings, answer: Answer, question: Question
    ) -> TransformResult:
        text = str(answer)
        output_question = self.output_question(dict(settings), question, {})
        try:
            candidates = _candidates(settings, text, question)
        except TimeoutError:
            return Unreadable(text, PATTERN_TIMEOUT_MESSAGE)
        if not candidates or (settings.occurrence == "only" and len(candidates) > 1):
            return Unreadable(text, _extraction_failure_reason(settings, candidates))
        found = candidates[-1] if settings.occurrence == "last" else candidates[0]
        if _is_blank_marker(found, question):
            return None
        return output_question.read(found)


def _is_blank_marker(text: str, question: Question) -> bool:
    """Whether ``text`` is the question's blank marker as its answers are read."""
    reading = question.reading
    return isinstance(reading, TextReading) and text == reading.normalized(reading.blank_marker)


def _candidates(settings: ExtractSettings, text: str, question: Question) -> list[str]:
    """What ``text``, an answer to ``question``, holds of the setting's type, in order."""
    regions = (
        [text] if settings.pattern is None else pattern_captures(settings.pattern, text, question)
    )
    if settings.type == "text":
        return regions
    numbers = (
        match[0].replace(",", "") for region in regions for match in _NUMBER.finditer(region)
    )
    return [number for number in numbers if reads_as_number(number, question.reading)]


def _extraction_failure_reason(settings: ExtractSettings, candidates: list[str]) -> str:
    if candidates:
        return f"Found several: {plain_join(candidates, conjunction='and')}."
    return "Couldn't find a number." if settings.type == "number" else "Couldn't find the text."
