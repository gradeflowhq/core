"""Regular expressions people write, run on answers students write."""

from typing import Annotated

import regex
from pydantic import AfterValidator, Field

from ...questions.base import Question
from ...questions.text import TextQuestion

PATTERN_TIME_LIMIT_S = 1.0
"""How long one pattern may spend on one answer: a pattern can backtrack for ever on some text."""
PATTERN_TIMEOUT_MESSAGE = "The pattern took too long on this answer."


def compile_pattern(pattern: str, flags: int = 0) -> regex.Pattern[str]:
    """``pattern`` compiled. Raises ``ValueError`` if it isn't valid."""
    try:
        return regex.compile(pattern, flags)
    except regex.error as error:
        raise ValueError(f"Not a valid pattern: {error}.") from None


def _check_compiles(pattern: str) -> str:
    compile_pattern(pattern)
    return pattern


def _check_one_group(pattern: str) -> str:
    if compile_pattern(pattern).groups != 1:
        raise ValueError("Write a pattern with one group, in brackets.")
    return pattern


_REGEX_FORMAT = Field(json_schema_extra={"format": "regex"})
"""A string's JSON Schema format when it is a regular expression."""

ValidPattern = Annotated[str, AfterValidator(_check_compiles), _REGEX_FORMAT]
"""A pattern that compiles, checked by its type, which a setting that fills one takes on."""
OneGroupPattern = Annotated[str, AfterValidator(_check_one_group), _REGEX_FORMAT]
"""A pattern with one group, around the part to find."""


def _compiled_for(pattern: str, question: Question) -> regex.Pattern[str]:
    # Answers to a question that ignores case are lowercased, so the pattern ignores it too.
    ignores_case = isinstance(question, TextQuestion) and question.reading.ignore_case
    return compile_pattern(pattern, regex.IGNORECASE if ignores_case else 0)


def pattern_matches(pattern: str, answer_text: str, question: Question) -> bool:
    """Whether ``pattern`` is found in ``answer_text``, an answer to ``question``. Raises
    ``TimeoutError`` if that takes too long."""
    return (
        _compiled_for(pattern, question).search(answer_text, timeout=PATTERN_TIME_LIMIT_S)
        is not None
    )


def pattern_capture_ranges(
    pattern: str, answer_text: str, question: Question
) -> list[tuple[int, int]]:
    """The start and end of each non-empty capture of the group of ``pattern`` in
    ``answer_text``, an answer to ``question``. Raises ``TimeoutError`` if that takes too long."""
    matches = _compiled_for(pattern, question).finditer(answer_text, timeout=PATTERN_TIME_LIMIT_S)
    return [(start, end) for start, end in (match.span(1) for match in matches) if start < end]
