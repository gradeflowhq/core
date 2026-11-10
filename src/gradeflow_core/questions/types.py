from dataclasses import dataclass
from typing import Literal, get_args

QuestionId = str
QuestionType = Literal["text", "number", "choice", "values", "code"]
ALL_QUESTION_TYPES: frozenset[QuestionType] = frozenset(get_args(QuestionType))
ValueType = Literal["text", "number"]

Value = str | int | float | None
"""A text or number answer, or one value of a multiple-values answer; None when blank."""
Answer = Value | frozenset[str] | list[Value]
"""What a question reads from an answer: a choice answer is its set of options."""


@dataclass(frozen=True)
class Unreadable:
    """An answer the question couldn't read, and why."""

    reason: str
