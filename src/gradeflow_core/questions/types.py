from dataclasses import dataclass
from typing import Literal, get_args

QuestionId = str
QuestionType = Literal["text", "number", "choice", "multipart", "list", "code"]
ALL_QUESTION_TYPES: frozenset[QuestionType] = frozenset(get_args(QuestionType))
ValueType = Literal["text", "number"]

Value = str | int | float | None
"""A text or number answer, or one part or item of an answer; None when blank."""
PartValues = dict[str, Value]
"""A multi-part answer, or a list item in parts: each part's value by name."""
Answer = Value | frozenset[str] | PartValues | list[Value] | list[PartValues]
"""What a question reads from an answer: a choice answer is its set of options, a list answer its
items."""


@dataclass(frozen=True)
class Unreadable:
    """An answer the question couldn't read, and why."""

    reason: str
    marked: bool = False
    """Whether it counts as marked, with no marks, instead of being left to a person."""
