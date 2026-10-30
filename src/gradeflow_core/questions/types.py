from dataclasses import dataclass
from typing import Literal, get_args

QuestionId = str
QuestionType = Literal["text", "number", "choice", "values", "code"]
ALL_TYPES: frozenset[QuestionType] = frozenset(get_args(QuestionType))
ValueType = Literal["text", "number"]
AnswerShape = Literal["single", "multiple"]
"""Whether a question reads each answer as one, or as several possible answers, the likeliest
first."""

Value = str | int | float | None
"""A text or number answer, or one value of a multiple-values answer; None when blank."""
Answer = Value | frozenset[str] | list[Value] | list[str]
"""What a question reads from an answer: a choice answer is its set of options, and an answer read
as several is its possible answers, the likeliest first."""


@dataclass(frozen=True)
class Unreadable:
    """An answer the question couldn't read, as written, and why."""

    raw: str
    reason: str
