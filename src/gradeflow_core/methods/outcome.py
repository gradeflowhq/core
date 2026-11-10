from dataclasses import dataclass
from typing import Literal, TypedDict

from ..questions.base import Question
from ..questions.types import Answer, QuestionId, Unreadable

FULL_MARKS_PROPORTION = 1 - 1e-9
"""A proportion this close to 1 is full marks, so rounding in sums doesn't cost a correct answer."""
NO_MARKS_PROPORTION = 1e-9
"""A proportion this close to 0 is no marks, so 0.1 + 0.2 - 0.3 in a sum gives none."""
ROUNDING_TOLERANCE = 1e-6
"""Marks this close count as equal."""

type Effect = Literal[
    "earned", "partial", "no_marks", "penalty", "uncounted", "met", "not_met", "unmarked"
]
"""What a check did to the marks. A check of no weight says whether its condition is met instead,
and "unmarked" leaves the answer to a person."""


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
    method_label: str = ""
    """The label of the method that gave it."""
    marked: bool = True
    name: str | None = None
    """The name of the check that gave it, if it has one."""
    parts: tuple[Outcome, ...] = ()
    """The outcomes of the check's own checks, in order."""
    is_skipped: bool = False
    """Its condition was false."""
    model_names: tuple[str, ...] = ()
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

    def first_error(self) -> str | None:
        """The first error in its details or those of its parts, such as why code couldn't run."""
        errors = (detail.value for detail in self.details if detail.kind == "error")
        return next(errors, None) or next(
            filter(None, (part.first_error() for part in self.parts)), None
        )

    def all_model_names(self) -> tuple[str, ...]:
        """The language models that decided it or any of its parts, each once."""
        return tuple(
            dict.fromkeys(
                (
                    *self.model_names,
                    *(model for part in self.parts for model in part.all_model_names()),
                )
            )
        )

    @property
    def correct(self) -> bool:
        return self.proportion >= FULL_MARKS_PROPORTION

    def template_values(self, question_id: str) -> PartResult:
        """The outcome as templates see it."""
        return {
            "name": self.name or self.method_label,
            "correct": self.correct,
            "proportion": self.proportion,
            "feedback": self.feedback,
            "question": question_id,
            "marked": self.marked,
            "skipped": self.is_skipped,
        }


def effect_of(outcome: Outcome) -> Effect:
    """What ``outcome`` does to the marks it counts towards."""
    if not outcome.marked:
        return "unmarked"
    return "earned" if outcome.correct else "partial" if outcome.proportion > 0 else "no_marks"


CORRECT_FEEDBACK = "Correct."
NO_ANSWER_FEEDBACK = "No answer."


def unanswered_outcome(answer: Answer | Unreadable) -> Outcome | None:
    """The outcome of a blank or unreadable answer, which no method marks; None for any other."""
    if isinstance(answer, Unreadable):
        return Outcome(0.0, answer.reason, marked=False)
    return Outcome(0.0, NO_ANSWER_FEEDBACK) if answer is None else None


def clamped(proportion: float) -> float:
    """``proportion`` kept between 0 and 1, and 0 when it is only rounding away from it."""
    return 0.0 if proportion < NO_MARKS_PROPORTION else min(proportion, 1.0)


class CheckOutput(TypedDict):
    """A check's output as later conditions and templates read it, as ``steps.<id>.output``."""

    correct: bool
    proportion: float
    feedback: str
    marked: bool
    """False when it couldn't mark an answer, so what it decides can't be marked either."""


class PartResult(CheckOutput):
    """One of a check's parts as its feedback reads it, as an item of ``parts``."""

    name: str
    question: str
    skipped: bool


type QuestionOutcomes = dict[QuestionId, Outcome]
type TransformResult = Answer | Unreadable


@dataclass(frozen=True)
class SkippedStep:
    """What a workflow gives for the output of a step that didn't run."""


type MethodResult = QuestionOutcomes | TransformResult | SkippedStep


@dataclass(frozen=True)
class StepResult:
    """An earlier step as later conditions and templates read it, as ``steps.<id>``."""

    value: CheckOutput | TransformResult
    skipped: bool = False
    """Its condition was false."""
    question: Question | None = None
    """The question that reads a transform's value."""

    @property
    def output(self) -> CheckOutput | Answer:
        return None if isinstance(self.value, Unreadable) else self.value

    @property
    def short_circuit_outcome(self) -> Outcome | None:
        """The outcome of a condition reading it, when it has nothing to test."""
        match self.value:
            case Unreadable(reason=reason):
                return Outcome(0.0, reason, marked=False)
            case None:
                return Outcome(0.0, is_skipped=True)
            case {"marked": False, "feedback": str(feedback)}:
                return Outcome(0.0, feedback, marked=False)
        return None

    @property
    def transform_result(self) -> TransformResult | SkippedStep:
        """What a later step reads of it: a check has no answer to read."""
        return SkippedStep() if self.skipped or isinstance(self.value, dict) else self.value
