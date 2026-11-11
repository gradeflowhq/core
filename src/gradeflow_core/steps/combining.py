"""How the outcomes of several weighted checks become one: a combination for each way of giving
marks."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from ..common.paths import pointer
from ..common.problems import Problem
from ..common.text import as_sentence, format_number, plain_markdown
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from .outcome import (
    CORRECT_FEEDBACK,
    FULL_MARKS_PROPORTION,
    Effect,
    Outcome,
    PartOutcome,
    QuestionOutcomes,
    Skipped,
    clamped,
    effect_of,
)

MarksBy = Literal["all_correct", "any_correct", "none_correct", "total", "proportional"]


@dataclass(frozen=True)
class Part:
    """One of the checks whose outcomes are combined, as combining needs to know it."""

    weight: float
    name: str | None = None
    """The name its author gave it."""
    default_name: str | None = None
    """What feedback calls it without one, such as "Value 2"."""
    method_label: str = ""
    """What templates call it without either."""
    has_own_feedback: bool = False
    """Whether its author wrote its feedback."""
    describe: Callable[[], str] = lambda: ""
    """What it checks, in markdown."""

    @property
    def feedback_name(self) -> str | None:
        return self.name or self.default_name

    @property
    def template_name(self) -> str:
        return self.feedback_name or self.method_label


type PartAndOutcome = tuple[Part, Outcome]
"""A part and its outcome for one answer."""


class Combination(ABC):
    """One way of giving marks for several checks."""

    label: str
    heading: str
    """How a description of the checks opens."""
    uses_weights = True

    def description_heading(self, *notes: str) -> str:
        """How a description of combined checks opens: ``Each one counts, in any order:``."""
        return f"{', '.join([self.heading, *notes])}:"

    def weight_notes(self, weight: float) -> tuple[str, ...]:
        """What ``weight``, when it isn't 1, makes a check worth."""
        return ()

    def weight_problems(self, weights: Sequence[float], setting_name: str) -> Iterator[Problem]:
        """A problem at each weight other than 0 and 1, when weights aren't used."""
        if self.uses_weights:
            return
        for index, weight in enumerate(weights):
            if weight not in (0, 1):
                yield Problem(
                    f"“{self.label}” doesn't use weights.",
                    field=pointer(setting_name, index, "weight"),
                )

    def denominator(self, weights: Sequence[float], max_marks: float) -> float:
        """What added marks are divided by, given the ``weights`` of every check counted."""
        return 1.0

    @abstractmethod
    def proportion(
        self, weighted_proportions: Sequence[tuple[float, float]], denominator: float
    ) -> float:
        """The proportion of ``(weight, proportion)`` parts of one answer combined, clamped."""

    def proportion_and_parts(
        self, parts_and_outcomes: Sequence[PartAndOutcome], max_marks: float
    ) -> tuple[float, tuple[PartOutcome, ...]]:
        """The combined proportion of one answer's outcomes, and each with what it did to the
        marks."""
        denominator = self.denominator([part.weight for part, _ in parts_and_outcomes], max_marks)
        proportion = self.proportion(
            [(part.weight, outcome.proportion) for part, outcome in parts_and_outcomes], denominator
        )
        return proportion, self._part_outcomes(parts_and_outcomes, denominator)

    @abstractmethod
    def combined_by_question(
        self, checks: Sequence[tuple[Part, Mapping[QuestionId, Outcome]]], questions: QuestionSet
    ) -> dict[QuestionId, Outcome]:
        """One outcome for each question, from the outcomes each of ``checks`` gave."""

    def _part_outcomes(
        self,
        parts_and_outcomes: Sequence[PartAndOutcome],
        denominator: float,
        counted_index: int | None = None,
    ) -> tuple[PartOutcome, ...]:
        """Each outcome with what it did to the marks. Only the one at ``counted_index``, if there
        is one, counts."""
        return tuple(
            PartOutcome(outcome, part.template_name, "uncounted")
            if counted_index is not None and index != counted_index
            else PartOutcome(
                outcome,
                part.template_name,
                self._effect(part.weight, outcome),
                self._contribution(part.weight, outcome, denominator),
            )
            for index, (part, outcome) in enumerate(parts_and_outcomes)
        )

    def _effect(self, weight: float, outcome: Outcome) -> Effect:
        return effect_of(outcome)

    def _contribution(self, weight: float, outcome: Outcome, denominator: float) -> float | None:
        return None


class _QuestionByQuestion(Combination):
    """A combination that marks each question from its own outcomes."""

    def combined_by_question(
        self, checks: Sequence[tuple[Part, Mapping[QuestionId, Outcome]]], questions: QuestionSet
    ) -> dict[QuestionId, Outcome]:
        return {
            question_id: self._combined(parts_and_outcomes, _max_marks(questions, question_id))
            for question_id, parts_and_outcomes in _outcomes_by_question(checks).items()
        }

    def _combined(self, parts_and_outcomes: Sequence[PartAndOutcome], max_marks: float) -> Outcome:
        proportion, part_outcomes = self.proportion_and_parts(parts_and_outcomes, max_marks)
        return Outcome(
            proportion,
            self._feedback(parts_and_outcomes, proportion >= FULL_MARKS_PROPORTION),
            marked=all(outcome.marked for _, outcome in parts_and_outcomes),
            parts=part_outcomes,
        )

    @abstractmethod
    def _feedback(self, parts_and_outcomes: Sequence[PartAndOutcome], is_correct: bool) -> str:
        """Why the answer got its combined proportion."""


class _AllCorrect(_QuestionByQuestion):
    label = "All must be correct"
    heading = "All of these"
    uses_weights = False

    def proportion(
        self, weighted_proportions: Sequence[tuple[float, float]], denominator: float
    ) -> float:
        return clamped(min(proportion for _, proportion in weighted_proportions))

    def _feedback(self, parts_and_outcomes: Sequence[PartAndOutcome], is_correct: bool) -> str:
        """What is correct of a correct answer, or else what isn't; and why any of it is left for a
        person."""
        return _joined_feedback(
            (part.feedback_name, outcome.feedback)
            for part, outcome in parts_and_outcomes
            if not outcome.marked or outcome.correct == is_correct
        )


class _NoneCorrect(_QuestionByQuestion):
    label = "None may be correct"
    heading = "None of these"
    uses_weights = False

    def proportion(
        self, weighted_proportions: Sequence[tuple[float, float]], denominator: float
    ) -> float:
        return clamped(1 - max(proportion for _, proportion in weighted_proportions))

    def _combined(self, parts_and_outcomes: Sequence[PartAndOutcome], max_marks: float) -> Outcome:
        outcome = super()._combined(parts_and_outcomes, max_marks)
        return replace(outcome, proportion=0.0) if _first_error(parts_and_outcomes) else outcome

    def _feedback(self, parts_and_outcomes: Sequence[PartAndOutcome], is_correct: bool) -> str:
        """Why parts weren't marked, and what was found, or "Correct."."""
        unmarked_feedback = _joined_feedback(
            (part.feedback_name, outcome.feedback)
            for part, outcome in parts_and_outcomes
            if not outcome.marked
        )
        not_allowed_reasons = [
            outcome.feedback
            if part.has_own_feedback
            else as_sentence(plain_markdown(part.describe()))
            for part, outcome in parts_and_outcomes
            if outcome.marked and outcome.proportion > 0
        ]
        if error := _first_error(parts_and_outcomes):
            not_allowed_line = as_sentence(f"Couldn't check the answer: {error}")
        else:
            not_allowed_line = (
                f"Not allowed. {' '.join(not_allowed_reasons)}" if not_allowed_reasons else ""
            )
        return (
            "\n".join(line for line in (unmarked_feedback, not_allowed_line) if line)
            or CORRECT_FEEDBACK
        )

    def _effect(self, weight: float, outcome: Outcome) -> Effect:
        return _penalty_effect(weight, outcome) if outcome.marked else effect_of(outcome)


class _AnyCorrect(Combination):
    label = "Any one must be correct"
    heading = "Any one of these"

    def weight_notes(self, weight: float) -> tuple[str, ...]:
        return () if weight == 1 else (f"{weight:.0%} of the marks",)

    def proportion(
        self, weighted_proportions: Sequence[tuple[float, float]], denominator: float
    ) -> float:
        return clamped(max(weight * proportion for weight, proportion in weighted_proportions))

    def proportion_and_parts(
        self, parts_and_outcomes: Sequence[PartAndOutcome], max_marks: float
    ) -> tuple[float, tuple[PartOutcome, ...]]:
        weighted_proportions = [
            (part.weight, outcome.proportion) for part, outcome in parts_and_outcomes
        ]
        proportion = self.proportion(weighted_proportions, 1.0)
        best_index = (
            max(
                range(len(weighted_proportions)),
                key=lambda index: weighted_proportions[index][0] * weighted_proportions[index][1],
            )
            if proportion > 0
            else None
        )
        return proportion, self._part_outcomes(parts_and_outcomes, 1.0, best_index)

    def combined_by_question(
        self, checks: Sequence[tuple[Part, Mapping[QuestionId, Outcome]]], questions: QuestionSet
    ) -> dict[QuestionId, Outcome]:
        """The outcomes of the check giving the most marks over the questions it marks, the first
        on a tie. Questions only other checks mark get no outcome."""

        def weighted_marks(check: tuple[Part, Mapping[QuestionId, Outcome]]) -> float:
            part, outcomes = check
            return sum(
                part.weight * outcome.proportion * _max_marks(questions, question_id)
                for question_id, outcome in outcomes.items()
            )

        best_part, best_outcomes = max(checks, key=weighted_marks)
        outcomes_by_question = _outcomes_by_question(checks)
        return {
            question_id: self._best_outcome(best_part, outcome, outcomes_by_question[question_id])
            for question_id, outcome in best_outcomes.items()
        }

    def _best_outcome(
        self, best_part: Part, best_outcome: Outcome, parts_and_outcomes: Sequence[PartAndOutcome]
    ) -> Outcome:
        """One answer's outcome from the best check's. It stays unmarked while a check that
        couldn't mark it could give more."""
        proportion = clamped(best_part.weight * best_outcome.proportion)
        unmarked_worth_more = [
            (part.feedback_name, outcome.feedback)
            for part, outcome in parts_and_outcomes
            if part is not best_part and not outcome.marked and part.weight > proportion
        ]
        feedback_lines = (
            _joined_feedback(unmarked_worth_more),
            _best_feedback(best_part, best_outcome),
        )
        best_index = next(
            index for index, (part, _) in enumerate(parts_and_outcomes) if part is best_part
        )
        return Outcome(
            proportion,
            "\n".join(line for line in feedback_lines if line),
            marked=best_outcome.marked and not unmarked_worth_more,
            parts=self._part_outcomes(
                parts_and_outcomes, 1.0, best_index if proportion > 0 else None
            ),
        )


@dataclass(frozen=True)
class _Additive(_QuestionByQuestion):
    """A combination that adds the parts' weighted marks up."""

    label: str
    heading: str
    divides_by_max_marks: bool
    """Whether weights are marks of the question, not shares of them."""

    def weight_notes(self, weight: float) -> tuple[str, ...]:
        return () if weight == 1 else (f"weight {format_number(weight)}",)

    def denominator(self, weights: Sequence[float], max_marks: float) -> float:
        if self.divides_by_max_marks:
            return max_marks or 1
        return sum(weight for weight in weights if weight > 0) or 1

    def proportion(
        self, weighted_proportions: Sequence[tuple[float, float]], denominator: float
    ) -> float:
        return clamped(
            sum(weight * proportion for weight, proportion in weighted_proportions) / denominator
        )

    def _feedback(self, parts_and_outcomes: Sequence[PartAndOutcome], is_correct: bool) -> str:
        """What gave marks and, unless the whole is correct, what took them off or fell short. A
        penalty says only that it applies, unless its feedback is its own."""
        lines: list[tuple[str | None, str]] = []
        for part, outcome in parts_and_outcomes:
            weighted_proportion = part.weight * outcome.proportion
            falls_short = weighted_proportion < 0 or (part.weight > 0 and not outcome.correct)
            if not outcome.marked:  # says why the whole is left for a person
                lines.append((part.feedback_name, outcome.feedback))
            elif weighted_proportion > 0 or (falls_short and not is_correct):
                if part.weight < 0 and not part.has_own_feedback:
                    lines.append((None, f"Penalty: {part.name}." if part.name else "Penalty."))
                else:
                    lines.append((part.feedback_name, outcome.feedback))
        return _joined_feedback(lines)

    def _effect(self, weight: float, outcome: Outcome) -> Effect:
        """With no weight or a negative one, a check that finds what it looks for is a penalty."""
        if not outcome.marked or weight > 0:
            return effect_of(outcome)
        return _penalty_effect(weight, outcome)

    def _contribution(self, weight: float, outcome: Outcome, denominator: float) -> float | None:
        return weight * outcome.proportion / denominator


COMBINATIONS: Mapping[MarksBy, Combination] = {
    "all_correct": _AllCorrect(),
    "any_correct": _AnyCorrect(),
    "none_correct": _NoneCorrect(),
    "total": _Additive("Total", "Total of these", divides_by_max_marks=True),
    "proportional": _Additive("Proportional", "Each one counts", divides_by_max_marks=False),
}
MARKS_BY_LABELS: dict[str, str] = {
    marks_by: combination.label for marks_by, combination in COMBINATIONS.items()
}


def combine(
    marks_by: MarksBy, checks: Sequence[tuple[Part, QuestionOutcomes]], questions: QuestionSet
) -> dict[QuestionId, Outcome | Skipped]:
    """One outcome for each question ``checks`` mark, combined as ``marks_by`` says. A question
    whose checks were all skipped is skipped."""
    weighted_checks = [(part, results) for part, results in checks if part.weight != 0]
    checks_run = [
        (part, outcomes)
        for part, results in weighted_checks
        if (
            outcomes := {
                question_id: result
                for question_id, result in results.items()
                if isinstance(result, Outcome)
            }
        )
    ]
    if not checks_run:
        return {question_id: Skipped() for _, results in weighted_checks for question_id in results}
    combined_outcomes = COMBINATIONS[marks_by].combined_by_question(checks_run, questions)
    return {
        question_id: replace(outcome, children=_children(checks, question_id, outcome.parts))
        for question_id, outcome in combined_outcomes.items()
    }


def _children(
    checks: Sequence[tuple[Part, QuestionOutcomes]],
    question_id: QuestionId,
    part_outcomes: tuple[PartOutcome, ...],
) -> tuple[PartOutcome, ...]:
    """The checks that ran for ``question_id``: those that count, which ``part_outcomes`` holds,
    and those of no weight, each with whether its condition is met."""
    counted_parts = iter(part_outcomes)
    return tuple(
        next(counted_parts)
        if part.weight != 0
        else PartOutcome(outcome, part.template_name, _condition_effect(outcome))
        for part, results in checks
        if (outcome := _outcome_for_question(part, results, question_id)) is not None
    )


def _outcome_for_question(
    part: Part, results: QuestionOutcomes, question_id: QuestionId
) -> Outcome | None:
    """A check's outcome for ``question_id``, or a no-weight check's outcome for the one question it
    reads; None if it was skipped."""
    result = results.get(question_id)
    if result is None and part.weight == 0 and len(results) == 1:
        [(other_question_id, result)] = results.items()
        if isinstance(result, Outcome):
            return replace(result, labels=(*result.labels, other_question_id))
    return result if isinstance(result, Outcome) else None


def _condition_effect(outcome: Outcome) -> Effect:
    if not outcome.marked:
        return "unmarked"
    return "met" if outcome.correct else "not_met"


def _penalty_effect(weight: float, outcome: Outcome) -> Effect:
    return "penalty" if outcome.proportion > 0 and weight != 0 else "uncounted"


def _outcomes_by_question(
    checks: Sequence[tuple[Part, Mapping[QuestionId, Outcome]]],
) -> dict[QuestionId, list[PartAndOutcome]]:
    outcomes_by_question: dict[QuestionId, list[PartAndOutcome]] = {}
    for part, outcomes in checks:
        for question_id, outcome in outcomes.items():
            outcomes_by_question.setdefault(question_id, []).append((part, outcome))
    return outcomes_by_question


def _max_marks(questions: QuestionSet, question_id: QuestionId) -> float:
    question = questions.get(question_id)
    return question.max_marks if question else 1.0


def _first_error(parts_and_outcomes: Sequence[PartAndOutcome]) -> str | None:
    return next(filter(None, (outcome.first_error() for _, outcome in parts_and_outcomes)), None)


def _best_feedback(best_part: Part, outcome: Outcome) -> str:
    """A correct check worth part of the marks, such as "Sign error", says what earned them better
    than its method's own "Correct.", unless its author wrote its feedback. Otherwise the best
    check speaks for the answer, without its name."""
    if outcome.proportion == 1 and 0 < best_part.weight < 1 and not best_part.has_own_feedback:
        name = best_part.feedback_name or "Partly correct"
        return f"{name} ({best_part.weight:.0%} of the marks)."
    return outcome.feedback


def _joined_feedback(named_feedback: Iterable[tuple[str | None, str]]) -> str:
    """Each feedback after the name of the check that gave it, each line once. A bare "Correct."
    adds nothing to other lines."""
    lines = list(
        dict.fromkeys(
            _with_check_name(name, feedback) for name, feedback in named_feedback if feedback
        )
    )
    lines_besides_correct = [line for line in lines if line != CORRECT_FEEDBACK]
    return "\n".join(lines_besides_correct or lines)


def _with_check_name(name: str | None, feedback: str) -> str:
    """Feedback after the name of the check that gave it, unless it only says the name again, as
    a check called "Correct" saying "Correct." does."""
    if not name or feedback.rstrip(".").casefold() == name.rstrip(".").casefold():
        return feedback
    return f"{name}: {feedback}"
