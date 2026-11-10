import re
import string
from collections.abc import Iterator

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import code_span, quoted
from ...ml import nli
from ...questions.base import Question
from ...questions.types import Answer
from ..matching import (
    MATCH_FIELD,
    SPELLING_CUTOFFS,
    STRICTNESS_FIELD,
    Match,
    Strictness,
    closest_spelling,
    match_note,
    match_problems,
    model_context,
)
from ..method import AnswerMethod
from ..outcome import Outcome
from ..position import Position
from ..settings import NonBlankText, Settings
from ..step import Step

# Entailment minus contradiction, the answer as premise: benchmarks/key-points-2026-10-26.
POINT_MEANING_CUTOFFS: dict[Strictness, float] = {"lenient": -0.05, "normal": 0.1, "strict": 0.9}


class KeyPointSettings(Settings):
    point: NonBlankText = Field(title="Point")
    match: Match = MATCH_FIELD
    strictness: Strictness = STRICTNESS_FIELD
    whole_words: bool = Field(
        default=False,
        title="Whole words only",
        description="So “for” doesn't match “format”.",
        json_schema_extra=hint(advanced=True, applies_when={"match": ["exact"]}),
    )


class KeyPoint(AnswerMethod[KeyPointSettings]):
    name = "gradeflow/key-point"
    version = 1
    label = "Key Point"
    summary = "Full marks if the answer includes one point."
    level = "internal"
    question_types = frozenset({"text"})
    settings = KeyPointSettings

    def describe(self, step: Step) -> str:
        settings = self._settings_of(step)
        whole_word_prefix = (
            "the whole word " if settings.whole_words and settings.match == "exact" else ""
        )
        point = code_span(settings.point)
        comparison_note = match_note(settings.match, settings.strictness)
        return f"Includes {whole_word_prefix}{point}{comparison_note}."

    def _mark_answer(
        self, settings: KeyPointSettings, answer: Answer, question: Question
    ) -> Outcome:
        matching_words = _matching_words(settings, str(answer), question)
        if matching_words is None:
            return Outcome(0.0, f"Missing {quoted(settings.point)}.")
        as_written_note = f" (as {quoted(matching_words)})" if matching_words else ""
        return Outcome(1.0, f"Has {quoted(settings.point)}{as_written_note}.")

    def problems(self, step: Step, position: Position) -> Iterator[Problem]:
        return match_problems(self._settings_of(step).match)


def _matching_words(settings: KeyPointSettings, answer: str, question: Question) -> str | None:
    """How the answer says the point: the words that match it, or empty when it says it in other
    words; None when it doesn't."""
    if settings.match == "meaning":
        cutoff = POINT_MEANING_CUTOFFS[settings.strictness]
        includes_point = nli.entails(answer, settings.point, model_context(question), cutoff)
        return "" if includes_point else None
    # The point is read as the answer was, so "CO2" still matches when capitals are ignored.
    point = str(question.read(settings.point) or settings.point)
    if settings.match == "spelling":
        point = " ".join(_words(point)) or point
        closest_words = _closest_words(point, answer, SPELLING_CUTOFFS[settings.strictness])
        return None if closest_words is None else "" if closest_words == point else closest_words
    pattern = re.escape(point)
    if settings.whole_words:
        pattern = rf"(?<!\w){pattern}(?!\w)"
    return "" if re.search(pattern, answer) else None


def _closest_words(point: str, answer: str, cutoff: float) -> str | None:
    """The run of the answer's words, as many as the point's, spelt most like it, if alike
    enough."""
    words = _words(answer)
    point_word_count = len(point.split())
    word_sequences = [
        " ".join(words[start : start + point_word_count])
        for start in range(len(words) - point_word_count + 1)
    ]
    best_index = closest_spelling(point, word_sequences, cutoff)
    return None if best_index is None else word_sequences[best_index]


def _words(text: str) -> list[str]:
    """The words of ``text``, without the punctuation around them."""
    return [word for word in (word.strip(string.punctuation) for word in text.split()) if word]
