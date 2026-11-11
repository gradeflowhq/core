from collections.abc import Mapping
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
    marked: bool = True
    """False when it couldn't mark the answer, which is left to a person."""
    details: tuple[Detail, ...] = ()
    model_names: tuple[str, ...] = ()
    """The language models that decided it."""
    parts: tuple[PartOutcome, ...] = ()
    """The outcomes it combines, in order."""
    title: str = ""
    """What the check that gave it checks, in a line of markdown."""
    labels: tuple[str, ...] = ()
    """What the title comes after: the check's name, and its question where that isn't the one
    marked."""
    children: tuple[PartOutcome, ...] = ()
    """What its breakdown shows under it."""

    def first_error(self) -> str | None:
        """The first error in its details or those of its parts, such as why code couldn't run."""
        errors = (detail.value for detail in self.details if detail.kind == "error")
        return next(errors, None) or next(
            filter(None, (part.outcome.first_error() for part in self.parts)), None
        )

    def all_model_names(self) -> tuple[str, ...]:
        """The language models that decided it or any of its parts, each once."""
        return tuple(
            dict.fromkeys(
                (
                    *self.model_names,
                    *(model for part in self.parts for model in part.outcome.all_model_names()),
                )
            )
        )

    @property
    def correct(self) -> bool:
        return self.proportion >= FULL_MARKS_PROPORTION


@dataclass(frozen=True)
class PartOutcome:
    """An outcome as part of the one that combines it."""

    outcome: Outcome
    name: str
    """What templates call the check that gave it."""
    effect: Effect
    """What it did to the marks of the outcome it is part of."""
    contribution: float | None = None
    """The part of that outcome's proportion it gave, when that outcome adds its parts up."""

    def template_values(self, question_id: str) -> PartResult:
        """The part as templates see it."""
        return {
            "name": self.name,
            "correct": self.outcome.correct,
            "proportion": self.outcome.proportion,
            "feedback": self.outcome.feedback,
            "question": question_id,
            "marked": self.outcome.marked,
            "skipped": False,
        }


@dataclass(frozen=True)
class Skipped:
    """What a step that didn't run gives: its condition was false, or what it reads didn't run."""

    feedback: str = ""
    """What the check that holds the steps says of an answer none of them marked."""


def outcome_of(result: Outcome | Skipped) -> Outcome:
    """``result`` as marks: a skipped check gives none."""
    return result if isinstance(result, Outcome) else Outcome(0.0, result.feedback)


def effect_of(outcome: Outcome) -> Effect:
    """What ``outcome`` does to the marks it counts towards."""
    if not outcome.marked:
        return "unmarked"
    return "earned" if outcome.correct else "partial" if outcome.proportion > 0 else "no_marks"


CORRECT_FEEDBACK = "Correct."
NO_ANSWER_FEEDBACK = "No answer."


def unanswered_outcome(answer: Unreadable | None) -> Outcome:
    """The outcome of an unreadable or blank answer, which no method marks."""
    if answer is None:
        return Outcome(0.0, NO_ANSWER_FEEDBACK)
    return Outcome(0.0, answer.reason, marked=False)


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


type QuestionOutcomes = Mapping[QuestionId, Outcome | Skipped]
"""What a check gives one student, by the question it marks."""
type TransformResult = Answer | Unreadable


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
    def condition_not_run_reason(self) -> Skipped | Unreadable | None:
        """What a condition reading it gives its step in place of running, when it has nothing to
        test."""
        match self.value:
            case Unreadable():
                return self.value
            case None:
                return Skipped()
            case {"marked": False, "feedback": str(feedback)}:
                return Unreadable(feedback)
        return None

    @property
    def transform_result(self) -> TransformResult | Skipped:
        """What a later step reads of it: a check has no answer to read."""
        return Skipped() if self.skipped or isinstance(self.value, dict) else self.value
