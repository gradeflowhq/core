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

type MatchMode = Literal["exact", "spelling", "meaning"]
type Strictness = Literal["lenient", "normal", "strict"]

SPELLING_CUTOFFS: dict[Strictness, float] = {"lenient": 0.7, "normal": 0.8, "strict": 0.9}
_MATCH_PHRASES: dict[MatchMode, str] = {
    "exact": "",
    "spelling": " in similar spelling",
    "meaning": " in meaning",
}


MATCH_FIELD: MatchMode = Field(
    default="exact",
    title="Match",
    description="Similar spelling accepts typos, and Same meaning accepts other words.",
    json_schema_extra=hint(
        labels={"exact": "Exactly", "spelling": "Similar spelling", "meaning": "Same meaning"}
    ),
)


STRICTNESS_FIELD: Strictness = Field(
    default="normal",
    title="How strict",
    json_schema_extra=hint(
        labels={"lenient": "Lenient", "normal": "Normal", "strict": "Strict"},
        applies_when={"match": ["spelling", "meaning"]},
    ),
)


def match_note(match: MatchMode, strictness: Strictness) -> str:
    """What a description says after the text matched: `` in similar spelling (strict)``."""
    strictness_note = f" ({strictness})" if match != "exact" and strictness != "normal" else ""
    return f"{_MATCH_PHRASES[match]}{strictness_note}"


def closest_spelling_index(text: str, candidates: Sequence[str], cutoff: float) -> int | None:
    """The index of the candidate spelt most like ``text``, the first on a tie, if alike enough."""
    closest_match = process.extractOne(text, candidates, scorer=Levenshtein.normalized_similarity)
    return closest_match[2] if closest_match and closest_match[1] >= cutoff else None


def model_context(question: Question) -> str:
    """What a language model reads of ``question`` before the texts it compares."""
    return (question.description or "").strip()


def match_problems(match: MatchMode) -> Iterator[Problem]:
    if match == "meaning" and not nli.is_available():
        yield Problem(MISSING_MODEL_MESSAGE, field=("match",))


TOLERANCE_FIELD = Field(
    default=0,
    ge=0,
    title="Tolerance",
    description="How far from the expected number an answer may be.",
    examples=[0.01, 0.05, 0.1],
)


def is_within_tolerance(answer: float, expected: float, tolerance: float) -> bool:
    """Whether ``answer`` is within ``tolerance`` of ``expected``, with slack for rounding."""
    return abs(answer - expected) <= tolerance + 1e-9 * max(abs(answer), abs(expected))


def expected_text(expected: str, tolerance: float) -> str:
    """``Expected about 12.57`` when there is a tolerance."""
    return f"Expected {'about ' if tolerance > 0 else ''}{expected}"


def unreadable_problems(
    texts: Sequence[str], read: Callable[[str], object], setting_name: str
) -> Iterator[Problem]:
    """Why each of ``texts``, the items of setting ``setting_name``, isn't read as an answer."""
    for index, text in enumerate(texts):
        try:
            answer = read(text)
        except ValueError as error:
            yield Problem(str(error), field=(setting_name, index))
            continue
        if answer is None:
            yield Problem("It reads as a blank answer.", field=(setting_name, index))


def duplicate_problems(
    texts: Sequence[str], read: Callable[[str], Hashable], setting_name: str
) -> Iterator[Problem]:
    """A problem at each item that reads the same as an earlier one."""
    first_item_numbers: dict[Hashable, int] = {}
    for index, text in enumerate(texts):
        if (answer := read_or_none(read, text)) is None:
            continue
        first_item_number = first_item_numbers.setdefault(answer, index + 1)
        if first_item_number != index + 1:
            yield Problem(
                f"It reads the same as item {first_item_number}.", field=(setting_name, index)
            )


def read_or_none[T](read: Callable[[str], T], text: str) -> T | None:
    """``text`` as ``read`` reads it; None if it can't."""
    try:
        return read(text)
    except ValueError:
        return None
