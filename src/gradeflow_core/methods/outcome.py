from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from ..questions.reading import write_answer
from ..questions.types import Answer, QuestionId, Unreadable

FULL_MARKS_PROPORTION = 1 - 1e-9
"""A proportion this close to 1 is full marks, so rounding in sums doesn't cost a correct answer."""
NO_MARKS_PROPORTION = 1e-9
"""A proportion this close to 0 is no marks, so 0.1 + 0.2 - 0.3 in a sum gives none."""
ROUNDING_TOLERANCE = 1e-6
"""Numbers this close count as equal, so rounding in sums such as 9.81 + 0.05 doesn't matter."""

type Effect = Literal[
    "earned", "partial", "no_marks", "penalty", "uncounted", "met", "not_met", "unmarked"
]
"""What a check did to the marks. A read-only check says whether its condition is met instead, and
"unmarked" leaves the answer to a person."""


@dataclass(frozen=True)
class Detail:
    """A fact about how a check marked an answer, such as what the code gave."""

    title: str
    value: str
    kind: Literal["text", "code", "error"] = "text"


@dataclass(frozen=True)
class Outcome:
    """What a check makes of an answer: the proportion it earns, from 0 to 1, and why."""

    proportion: float
    feedback: str = ""
    method: str = ""
    """The label of the method that gave it."""
    marked: bool = True
    name: str | None = None
    """The name of the check that gave it, if it has one."""
    parts: tuple[Outcome, ...] = ()
    """The outcomes of the check's own checks, in order."""
    skipped: bool = False
    """Its condition was false, so it did not execute or participate in aggregation."""
    models: tuple[str, ...] = ()
    """The language models that decided it."""
    title: str = ""
    """What the check that gave it checks, in a line of markdown."""
    labels: tuple[str, ...] = ()
    """What the title comes after: the check's name, and its question where that isn't the one
    marked."""
    details: tuple[Detail, ...] = ()
    children: tuple[Outcome, ...] = ()
    """Its children in the breakdown."""
    effect: Effect | None = None
    """What it did to the marks of the check it is part of."""
    contribution: float | None = None
    """The part of that check's proportion it gave, when that check adds its parts up."""

    def all_models(self) -> tuple[str, ...]:
        """The language models that decided it or any of its parts, each once."""
        return tuple(
            dict.fromkeys(
                (*self.models, *(model for part in self.parts for model in part.all_models()))
            )
        )

    @property
    def correct(self) -> bool:
        return self.proportion >= FULL_MARKS_PROPORTION

    def template_values(self, question: str) -> dict[str, Any]:
        """The outcome as templates see it."""
        return {
            "name": self.name or self.method,
            "correct": self.correct,
            "proportion": self.proportion,
            "feedback": self.feedback,
            "question": question,
            "marked": self.marked,
            "skipped": self.skipped,
        }


def effect_of(outcome: Outcome) -> Effect:
    """What ``outcome`` does to the marks it counts towards."""
    if not outcome.marked:
        return "unmarked"
    return "earned" if outcome.correct else "partial" if outcome.proportion > 0 else "no_marks"


CORRECT = "Correct."
NO_ANSWER = "No answer."


def unanswered_outcome(answer: Answer | Unreadable) -> Outcome | None:
    """The outcome of a blank or unreadable answer, which no method marks; None for any other."""
    if isinstance(answer, Unreadable):
        return Outcome(0.0, answer.reason, marked=False)
    return Outcome(0.0, NO_ANSWER) if answer is None else None


def clamped(proportion: float) -> float:
    """``proportion`` kept between 0 and 1, and 0 when it is only rounding away from it."""
    return 0.0 if proportion < NO_MARKS_PROPORTION else min(proportion, 1.0)


def feedback_scope(
    outcome: Outcome, answer: Answer | Unreadable, question: QuestionId
) -> dict[str, Any]:
    """What a feedback template sees of how a check marked ``answer``."""
    return {
        "answer": answer.answer if isinstance(answer, Unreadable) else write_answer(answer),
        "correct": outcome.correct,
        "proportion": outcome.proportion,
        "feedback": outcome.feedback,
        "parts": [part.template_values(question) for part in outcome.parts],
    }


class EarlierResult(TypedDict):
    """A check's result as later conditions and templates read it, as ``checks.<id>``."""

    correct: bool
    proportion: float
    feedback: str
    marked: bool
    """False when it couldn't mark an answer, so what it decides can't be marked either."""
    skipped: bool
    """True when its condition skipped it, distinct from a check awarding zero."""
