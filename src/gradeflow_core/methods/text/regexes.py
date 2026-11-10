"""Regular expressions people write, run on answers students write."""

import regex

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


def _compiled_for(pattern: str, question: Question) -> regex.Pattern[str]:
    # Answers to a question that ignores case are lowercased, so the pattern ignores it too.
    ignores_case = isinstance(question, TextQuestion) and question.reading.ignore_case
    return compile_pattern(pattern, regex.IGNORECASE if ignores_case else 0)


def pattern_matches(pattern: str, text: str, question: Question) -> bool:
    """Whether ``pattern`` is found in ``text``, an answer to ``question``. Raises
    ``TimeoutError`` if that takes too long."""
    return _compiled_for(pattern, question).search(text, timeout=PATTERN_TIME_LIMIT_S) is not None


def pattern_captures(pattern: str, text: str, question: Question) -> list[str]:
    """What the one group of ``pattern`` captures in ``text``, an answer to ``question``. Raises
    ``TimeoutError`` if that takes too long."""
    matches = _compiled_for(pattern, question).findall(text, timeout=PATTERN_TIME_LIMIT_S)
    return list(filter(None, matches))
