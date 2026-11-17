from collections.abc import Iterable, Mapping
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
class Span:
    """Characters from ``start`` (inclusive) to ``end`` (exclusive) of an answer as its question
    reads it, or of its value ``value_number``, from 1."""

    start: int
    end: int
    value_number: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.end:
            raise ValueError(
                f"A span needs 0 ≤ start ≤ end, but has start {self.start} and end {self.end}."
            )

    def subspan(self, span: Span) -> Span:
        """``span``, counted from this span's start, as a span of the text this span is in.
        Raises ``ValueError`` if it ends past this span."""
        if span.end > self.end - self.start:
            length = self.end - self.start
            raise ValueError(
                f"A subspan ends at {span.end}, past the {length} characters it is in."
            )
        return Span(self.start + span.start, self.start + span.end, self.value_number)


def merged(spans: Iterable[Span]) -> tuple[Span, ...]:
    """``spans`` sorted, without empty ones, and with those that overlap or touch merged."""
    result: list[Span] = []
    nonempty = (span for span in spans if span.start < span.end)
    for span in sorted(nonempty, key=lambda span: (span.value_number or 0, span.start)):
        last = result[-1] if result else None
        if last and last.value_number == span.value_number and span.start <= last.end:
            result[-1] = Span(last.start, max(last.end, span.end), span.value_number)
        else:
            result.append(span)
    return tuple(result)


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
    spans: tuple[Span, ...] = ()
    """The parts of the answer the transforms it read extracted their values from. Marking sets
    it, replacing any a method gives."""

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

    def all_spans(self) -> tuple[Span, ...]:
        """Its spans and its parts'."""
        return (*self.spans, *(span for part in self.parts for span in part.outcome.all_spans()))

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
    return Outcome(0.0, answer.reason, marked=answer.marked)


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


@dataclass(frozen=True)
class Transformed:
    """What a transform gives: another answer, and the parts of its input it extracted it from."""

    answer: Answer
    spans: tuple[Span, ...] = ()


type TransformResult = Transformed | Unreadable


@dataclass(frozen=True)
class StepResult:
    """An earlier step as later conditions and templates read it, as ``steps.<id>``."""

    value: CheckOutput | Answer | Unreadable
    skipped: bool = False
    """Its condition was false."""
    question: Question | None = None
    """The question that reads a transform's value."""
    spans: tuple[Span, ...] = ()
    """The parts of the answer to ``spans_question_id`` a transform extracted its value from."""
    spans_question_id: QuestionId | None = None
    """The question whose answer ``spans`` are in."""

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

    def spans_in(self, question_id: QuestionId | None) -> tuple[Span, ...]:
        """Its spans, if they are in the answer to ``question_id``."""
        return self.spans if question_id == self.spans_question_id else ()

    def transform_result(self, question_id: QuestionId | None) -> TransformResult | Skipped:
        """What a later step that decides the marks of ``question_id`` reads of it: its value,
        with its spans if they are in that question's answer. A check has no value to read."""
        if self.skipped or isinstance(self.value, dict):
            return Skipped()
        if isinstance(self.value, Unreadable):
            return self.value
        return Transformed(self.value, self.spans_in(question_id))
