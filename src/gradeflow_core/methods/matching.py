"""How methods match answers: the choices, cut-offs and tolerance they share, and reading the
answers they expect."""

from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

from pydantic import Field, JsonValue
from rapidfuzz.distance import Levenshtein

from ..common.hints import hint
from ..common.problems import Problem
from ..ml import MISSING_MODEL, nli
from ..questions.base import Question
from .outcome import ROUNDING_TOLERANCE

type Match = Literal["exact", "spelling", "meaning"]
type Strictness = Literal["lenient", "normal", "strict"]

MATCH_LABELS = {"exact": "Exactly", "spelling": "Similar spelling", "meaning": "Same meaning"}
STRICTNESS_LABELS = {"lenient": "Lenient", "normal": "Normal", "strict": "Strict"}
SPELLING_CUTOFFS: dict[Strictness, float] = {"lenient": 0.7, "normal": 0.8, "strict": 0.9}
_HOW: dict[Match, str] = {"exact": "", "spelling": " in similar spelling", "meaning": " in meaning"}


MATCH: Match = Field(
    default="exact",
    title="Match",
    description="Similar spelling forgives small mistakes, and Same meaning accepts other words.",
    json_schema_extra=hint(labels=MATCH_LABELS),
)


def strictness_field(applies_when: dict[str, JsonValue] | None = None) -> Any:
    """The How strict field, Normal by default."""
    return Field(
        default="normal",
        title="How strict",
        json_schema_extra=hint(labels=STRICTNESS_LABELS, applies_when=applies_when),
    )


def how_matched(match: Match, strictness: Strictness) -> str:
    """What a description says after the text matched: `` in similar spelling (strict)``."""
    loose = f" ({strictness})" if match != "exact" and strictness != "normal" else ""
    return f"{_HOW[match]}{loose}"


def most_alike(text: str, candidates: Sequence[str], cutoff: float) -> int | None:
    """The index of the candidate spelt most like ``text``, the first on a tie, if alike enough."""
    best = max(
        range(len(candidates)),
        key=lambda at: Levenshtein.normalized_similarity(text, candidates[at]),
        default=None,
    )
    if best is None or Levenshtein.normalized_similarity(text, candidates[best]) < cutoff:
        return None
    return best


def context_of(question: Question) -> str:
    """What a language model reads of ``question`` before the texts it compares."""
    return (question.description or "").strip()


def match_problems(match: Match) -> Iterator[Problem]:
    if match == "meaning" and not nli.available():
        yield Problem(MISSING_MODEL, field=("match",))


TOLERANCE = Field(
    default=0,
    ge=0,
    title="Tolerance",
    description="How far from the expected number an answer may be.",
    examples=[0.01, 0.05, 0.1],
)


def within(answer: float, expected: float, tolerance: float) -> bool:
    return abs(answer - expected) <= tolerance + ROUNDING_TOLERANCE


def expecting(expected: str, tolerance: float) -> str:
    """``Expected about 12.57`` when there is a tolerance."""
    return f"Expected {'about ' if tolerance > 0 else ''}{expected}"


def reading_problems(
    texts: Sequence[str], read: Callable[[str], object], field: str
) -> Iterator[Problem]:
    """Why each of ``texts``, the items of setting ``field``, isn't read as an answer."""
    for index, text in enumerate(texts):
        try:
            answer = read(text)
        except ValueError as error:
            yield Problem(str(error), field=(field, index))
            continue
        if answer is None:
            yield Problem("It reads as a blank answer.", field=(field, index))


def read_or_none[T](read: Callable[[str], T], text: str) -> T | None:
    """``text`` as ``read`` reads it; None if it can't."""
    try:
        return read(text)
    except ValueError:
        return None
