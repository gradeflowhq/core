"""How methods match answers: the choices, cut-offs and tolerance they share, and reading the
answers they expect."""

from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Literal

from pydantic import Field
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from ..common.hints import hint
from ..common.problems import Problem
from ..ml import MISSING_MODEL_MESSAGE, nli
from ..questions.base import Question

type Match = Literal["exact", "spelling", "meaning"]
type Strictness = Literal["lenient", "normal", "strict"]

MATCH_LABELS = {"exact": "Exactly", "spelling": "Similar spelling", "meaning": "Same meaning"}
STRICTNESS_LABELS = {"lenient": "Lenient", "normal": "Normal", "strict": "Strict"}
SPELLING_CUTOFFS: dict[Strictness, float] = {"lenient": 0.7, "normal": 0.8, "strict": 0.9}
_MATCH_PHRASES: dict[Match, str] = {
    "exact": "",
    "spelling": " in similar spelling",
    "meaning": " in meaning",
}


MATCH_FIELD: Match = Field(
    default="exact",
    title="Match",
    description="Similar spelling accepts typos, and Same meaning accepts other words.",
    json_schema_extra=hint(labels=MATCH_LABELS),
)


STRICTNESS_FIELD: Strictness = Field(
    default="normal",
    title="How strict",
    json_schema_extra=hint(
        labels=STRICTNESS_LABELS, applies_when={"match": ["spelling", "meaning"]}
    ),
)


def match_note(match: Match, strictness: Strictness) -> str:
    """What a description says after the text matched: `` in similar spelling (strict)``."""
    strictness_note = f" ({strictness})" if match != "exact" and strictness != "normal" else ""
    return f"{_MATCH_PHRASES[match]}{strictness_note}"


def closest_spelling(text: str, candidates: Sequence[str], cutoff: float) -> int | None:
    """The index of the candidate spelt most like ``text``, the first on a tie, if alike enough."""
    found = process.extractOne(text, candidates, scorer=Levenshtein.normalized_similarity)
    return found[2] if found and found[1] >= cutoff else None


def model_context(question: Question) -> str:
    """What a language model reads of ``question`` before the texts it compares."""
    return (question.description or "").strip()


def match_problems(match: Match) -> Iterator[Problem]:
    if match == "meaning" and not nli.available():
        yield Problem(MISSING_MODEL_MESSAGE, field=("match",))


TOLERANCE_FIELD = Field(
    default=0,
    ge=0,
    title="Tolerance",
    description="How far from the expected number an answer may be.",
    examples=[0.01, 0.05, 0.1],
)


def within(answer: float, expected: float, tolerance: float) -> bool:
    """Whether ``answer`` is within ``tolerance`` of ``expected``, with slack for rounding."""
    return abs(answer - expected) <= tolerance + 1e-9 * max(abs(answer), abs(expected))


def expected_text(expected: str, tolerance: float) -> str:
    """``Expected about 12.57`` when there is a tolerance."""
    return f"Expected {'about ' if tolerance > 0 else ''}{expected}"


def unreadable_problems(
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


def duplicate_problems(
    texts: Sequence[str], read: Callable[[str], Hashable], field: str
) -> Iterator[Problem]:
    """A problem at each item that reads the same as an earlier one."""
    first_item_numbers: dict[Hashable, int] = {}
    for index, text in enumerate(texts):
        if (value := read_or_none(read, text)) is None:
            continue
        first = first_item_numbers.setdefault(value, index + 1)
        if first != index + 1:
            yield Problem(f"It reads the same as item {first}.", field=(field, index))


def read_or_none[T](read: Callable[[str], T], text: str) -> T | None:
    """``text`` as ``read`` reads it; None if it can't."""
    try:
        return read(text)
    except ValueError:
        return None
