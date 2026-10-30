from dataclasses import dataclass, replace
from typing import Any, TypedDict

from ..questions.reading import write_answer
from ..questions.types import Answer, QuestionId, Unreadable

FULL_MARKS = 1 - 1e-9
"""A proportion this close to 1 is full marks, so rounding in sums doesn't cost a correct answer."""
ROUNDING_TOLERANCE = 1e-6
"""Numbers this close count as equal, so rounding in sums such as 9.81 + 0.05 doesn't matter."""


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

    def all_models(self) -> tuple[str, ...]:
        """The language models that decided it or any of its parts, each once."""
        return tuple(
            dict.fromkeys((*self.models, *(m for p in self.parts for m in p.all_models())))
        )

    @property
    def correct(self) -> bool:
        return self.proportion >= FULL_MARKS

    def labelled(self, method: str) -> Outcome:
        return self if self.method else replace(self, method=method)

    def view(self, question: str) -> dict[str, Any]:
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


CORRECT = "Correct."
NO_ANSWER = "No answer."


def unanswered(answer: Answer | Unreadable) -> Outcome | None:
    """The outcome of a blank or unreadable answer, which no method marks; None for any other."""
    if isinstance(answer, Unreadable):
        return Outcome(0.0, answer.reason, marked=False)
    return Outcome(0.0, NO_ANSWER) if answer is None else None


def clamped(proportion: float) -> float:
    """``proportion`` kept between 0 and 1."""
    return min(max(proportion, 0.0), 1.0)


def feedback_scope(
    outcome: Outcome, answer: Answer | Unreadable, question: QuestionId
) -> dict[str, Any]:
    """What a feedback template sees of how a check marked ``answer``."""
    return {
        "answer": answer.raw if isinstance(answer, Unreadable) else write_answer(answer),
        "correct": outcome.correct,
        "proportion": outcome.proportion,
        "feedback": outcome.feedback,
        "parts": [part.view(question) for part in outcome.parts],
    }


class CheckResult(TypedDict):
    """A check's result as later conditions and templates read it, as ``checks.<id>``."""

    correct: bool
    proportion: float
    feedback: str
    marked: bool
    """False when it couldn't mark an answer, so what it decides can't be marked either."""
    skipped: bool
    """True when its condition skipped it, distinct from a check awarding zero."""
