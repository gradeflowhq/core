import re
import string
from collections.abc import Iterator

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import markdown_code, quoted
from ...ml import nli
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..matching import (
    MATCH,
    SPELLING_CUTOFFS,
    Match,
    Strictness,
    context_of,
    how_matched,
    match_problems,
    most_alike,
    strictness_field,
)
from ..method import AnswerMethod
from ..outcome import Outcome
from ..position import Position
from ..settings import Settings, Text

# Entailment minus contradiction, the answer as premise: benchmarks/key-points-2026-10-26.
MEANING_CUTOFFS: dict[Strictness, float] = {"lenient": -0.05, "normal": 0.1, "strict": 0.9}


class KeyPointSettings(Settings):
    point: Text = Field(title="Point")
    match: Match = MATCH
    strictness: Strictness = strictness_field(applies_when={"match": ["spelling", "meaning"]})
    whole_words: bool = Field(
        default=False,
        title="Whole words only",
        description="So “for” doesn't match “format”.",
        json_schema_extra=hint(advanced=True, applies_when={"match": "exact"}),
    )


class KeyPoint(AnswerMethod[KeyPointSettings]):
    name = "gradeflow/key-point"
    version = 1
    label = "Key Point"
    summary = "Full marks if the answer includes one point."
    level = "internal"
    question_types = frozenset({"text"})
    settings = KeyPointSettings

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        whole = "the whole word " if settings.whole_words and settings.match == "exact" else ""
        point = markdown_code(settings.point)
        return f"Includes {whole}{point}{how_matched(settings.match, settings.strictness)}."

    def _mark_answer(
        self, settings: KeyPointSettings, answer: Answer, question: Question
    ) -> Outcome:
        found = _found(settings, str(answer), question)
        if found is None:
            return Outcome(0.0, f"Missing {quoted(settings.point)}.")
        written = f" (as {quoted(found)})" if found else ""
        return Outcome(1.0, f"Has {quoted(settings.point)}{written}.")

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        return match_problems(self._settings_of(check).match)


def _found(settings: KeyPointSettings, answer: str, question: Question) -> str | None:
    """How the answer says the point: the words that match it, or empty when it says it in other
    words; None when it doesn't."""
    if settings.match == "meaning":
        cutoff = MEANING_CUTOFFS[settings.strictness]
        said = nli.says(answer, settings.point, context_of(question), cutoff)
        return "" if said else None
    # The point is read as the answer was, so "CO2" still matches when capitals are ignored.
    point = str(question.read(settings.point) or settings.point)
    if settings.match == "spelling":
        found = _closest(point, answer, SPELLING_CUTOFFS[settings.strictness])
        return None if found is None else "" if found == point else found
    pattern = re.escape(point)
    if settings.whole_words:
        pattern = rf"(?<!\w){pattern}(?!\w)"
    return "" if re.search(pattern, answer) else None


def _closest(point: str, answer: str, cutoff: float) -> str | None:
    """The run of the answer's words, as many as the point's, spelt most like it, if alike
    enough."""
    words = [word for word in (word.strip(string.punctuation) for word in answer.split()) if word]
    size = len(point.split())
    runs = [" ".join(words[start : start + size]) for start in range(len(words) - size + 1)]
    best = most_alike(point, runs, cutoff)
    return None if best is None else runs[best]
