from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Annotated, Any, Literal

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import format_number, markdown_list, plain_markdown, sentence
from ...questions.question_set import QuestionSet
from ...questions.types import ALL_TYPES, QuestionId
from ..check import Check, Context, Marking
from ..example import examples_beside
from ..method import Method
from ..outcome import CORRECT, FULL_MARKS, CheckResult, Outcome, clamped
from ..position import Position
from ..settings import Settings

Combine = Literal["all", "any", "none", "total", "proportional"]
COMBINE_LABELS = {
    "all": "All must be correct",
    "any": "Any one must be correct",
    "none": "None may be correct",
    "total": "Total",
    "proportional": "Proportional",
}
CheckItem = Annotated[Marking, Field(title="Check", json_schema_extra=hint(kind="check"))]


class CombinationSettings(Settings):
    combine: Combine = Field(
        default="all",
        title="How to combine",
        json_schema_extra=hint(labels=COMBINE_LABELS),
    )
    checks: list[CheckItem] = Field(
        default_factory=list, title="Checks", description="The checks to combine."
    )


HEADINGS = {
    **COMBINE_LABELS,
    "total": "Marks added up",
    "proportional": "Each one counts",
}
"""How a combination's description opens."""

type Counted = list[tuple[Check, dict[QuestionId, Outcome]]]


def marked_in_order(
    checks: Sequence[Check],
    context: Context,
    question: QuestionId | None,
    values: Sequence[int | None],
) -> Counted:
    """Each check's outcomes, the check reading ``question`` at its value of ``values``; a check's
    condition sees the results of the earlier ones."""
    earlier: dict[str, CheckResult] = {}
    counted: Counted = []
    for check, value in zip(checks, values, strict=True):
        seeing = replace(context, earlier={**context.earlier, **earlier})
        outcomes = check.mark(seeing, question, value)
        if check.id:
            earlier[check.id] = _result_of(outcomes)
        counted.append((check, outcomes))
    return counted


def marked_by(checks: Sequence[Check], question: QuestionId | None) -> set[QuestionId]:
    """The questions ``checks`` give marks to, reading ``question`` unless they name another."""
    return {
        key for check in checks if not check.read_only for key in check.questions_marked(question)
    }


def read_by(checks: Sequence[Check], question: QuestionId | None) -> set[QuestionId]:
    """The questions whose answers ``checks`` read, reading ``question`` unless they name
    another."""
    return {key for check in checks for key in check.questions_read(question)}


def combine(
    settings: CombinationSettings, counted: Counted, questions: QuestionSet
) -> dict[QuestionId, Outcome]:
    """One outcome for each question the counted checks mark, combined as ``settings`` says. A
    question whose checks were all skipped is skipped."""
    counting = [(check, outcomes) for check, outcomes in counted if not check.read_only]
    active = [
        (check, kept)
        for check, outcomes in counting
        if (kept := {key: outcome for key, outcome in outcomes.items() if not outcome.skipped})
    ]
    if not active:
        keys = dict.fromkeys(key for _, outcomes in counting for key in outcomes)
        return {key: Outcome(0.0, skipped=True) for key in keys}
    if settings.combine == "any":
        return _best(active, questions)
    keys = dict.fromkeys(key for _, outcomes in active for key in outcomes)
    combined: dict[QuestionId, Outcome] = {}
    for key in keys:
        items = [(check, outcomes[key]) for check, outcomes in active if key in outcomes]
        parts = tuple(outcome for _, outcome in items)
        proportion = clamped(
            combined_proportion(
                settings.combine,
                [(check.weight, outcome.proportion) for check, outcome in items],
                [check.weight for check, _ in items],
                _max_marks(questions, key),
            )
        )
        feedback = (
            _not_allowed(items)
            if settings.combine == "none"
            else _feedback(_explaining(settings.combine, items, proportion >= FULL_MARKS))
        )
        combined[key] = Outcome(
            proportion,
            feedback,
            marked=all(p.marked for p in parts),
            parts=parts,
        )
    return combined


def _explaining(
    combine: Combine, items: list[tuple[Check, Outcome]], correct: bool
) -> tuple[Outcome, ...]:
    """The parts that explain the combined proportion: what counted for it, and, unless the whole
    is correct, what counted against it. Only "Total" and "Proportional" keep what counted for an
    incorrect whole, as each part earned its share. A penalty says only that it took marks off,
    unless its feedback is its own."""
    explaining = []
    for check, part in items:
        if not part.marked:  # says why the whole is left for a person
            explaining.append(part)
            continue
        adding = combine in ("total", "proportional")
        weight = check.weight if adding else 1
        match combine:
            case "all":
                gave, cost = part.correct, not part.correct
            case _:
                gave = weight * part.proportion > 0
                cost = weight * part.proportion < 0 or (weight > 0 and not part.correct)
        if (gave and (correct or adding)) or (cost and not correct):
            penalty = weight < 0 and not check.feedback
            taken_off = replace(part, feedback=_taken_off(check), name=None)
            explaining.append(taken_off if penalty else part)
    return tuple(explaining)


def _not_allowed(items: list[tuple[Check, Outcome]]) -> str:
    """Under "None may be correct": why parts are left for a person, and "Not allowed." with what
    each check that found the answer checks, unless its feedback is its own; else "Correct."."""
    found = [
        part.feedback if check.feedback else sentence(plain_markdown(check.describe()))
        for check, part in items
        if part.marked and part.proportion > 0
    ]
    unmarked = _feedback(tuple(part for _, part in items if not part.marked))
    said = f"Not allowed. {' '.join(found)}" if found else ""
    return "\n".join(line for line in (unmarked, said) if line) or CORRECT


def combined_proportion(
    combine: Combine,
    weighted: Sequence[tuple[float, float]],
    weights: Sequence[float],
    max_marks: float,
) -> float:
    """The proportion of ``(weight, proportion)`` parts of one answer combined, before clamping.
    A total is marks of a question of ``max_marks``; a proportional share is out of the positive
    ``weights`` of every check counted."""
    proportions = [proportion for _, proportion in weighted]
    earned = sum(weight * proportion for weight, proportion in weighted)
    match combine:
        case "all":
            return min(proportions)
        case "none":
            return 1 - max(proportions)
        case "any":
            return max(weight * proportion for weight, proportion in weighted)
        case "total":
            return earned / (max_marks or 1)
        case "proportional":
            return earned / (sum(weight for weight in weights if weight > 0) or 1)


def _best(counted: Counted, questions: QuestionSet) -> dict[QuestionId, Outcome]:
    """The outcomes of the check giving the most marks over the questions it marks; the first on
    a tie. Questions only other checks mark get no marks."""

    def total(item: tuple[Check, dict[QuestionId, Outcome]]) -> float:
        check, outcomes = item
        return sum(
            check.weight * o.proportion * _max_marks(questions, key) for key, o in outcomes.items()
        )

    best_check, best = max(counted, key=total)

    def spoken(outcome: Outcome) -> str:
        # A correct check worth part of the marks, such as "Sign error", says what earned them
        # better than its method's own "Correct.", unless its author wrote its feedback. Otherwise
        # the best check speaks for the answer, without its name.
        if outcome.proportion == 1 and 0 < best_check.weight < 1 and not best_check.feedback:
            return f"{outcome.name or 'Partly right'} ({best_check.weight:.0%} of the marks)."
        return outcome.feedback

    return {
        key: Outcome(
            clamped(best_check.weight * outcome.proportion),
            spoken(outcome),
            marked=outcome.marked,
            parts=tuple(outcomes[key] for _, outcomes in counted if key in outcomes),
        )
        for key, outcome in best.items()
    }


def _max_marks(questions: QuestionSet, key: QuestionId) -> float:
    question = questions.get(key)
    return question.max_marks if question else 1.0


def _feedback(parts: tuple[Outcome, ...]) -> str:
    """The parts' feedback, each line once. A bare "Correct." adds nothing to other lines."""
    lines = list(dict.fromkeys(_named(part.name, part.feedback) for part in parts if part.feedback))
    said = [line for line in lines if line != CORRECT]
    return "\n".join(said or lines)


def _named(name: str | None, feedback: str) -> str:
    """Feedback after the name of the check that gave it, unless it only says the name again, as
    a check called "Correct" saying "Correct." does."""
    if not name or feedback.rstrip(".").casefold() == name.rstrip(".").casefold():
        return feedback
    return f"{name}: {feedback}"


class Combination(Method[CombinationSettings]):
    name = "gradeflow/combination"
    version = 1
    label = "Combination"
    summary = "Several checks on one answer, combined into one mark."
    guide = (
        "**Use it when:** One check isn't enough, such as when an answer must name a key term and "
        "stay within a word limit, or when some answers deserve part of the marks.\n\n"
        "**How it marks:** Each check is marked, and **How to combine** sets how: **All must be "
        "correct** and **None may be correct** give full marks or none, **Any one must be "
        "correct** counts the check that gives the most marks, **Total** adds up the correct "
        "checks by their weights, as marks of the question, and **Proportional** gives each its "
        "share of the marks by its weight. A weight below 1, such as `0.5`, makes a check worth "
        "part of the marks, and a negative weight takes marks off.\n\n"
        "**Good to know:** With **Any one must be correct**, list the right answer first, then "
        "partly right answers with lower weights.\n\n"
        "**Related:** To forbid something, add a Not Allowed check. To mark several questions "
        "together, make a question group."
    )
    level = "advanced"
    question_types = ALL_TYPES
    answer_shape = "both"
    group = True
    settings = CombinationSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        checks = markdown_list(
            child.headed(_weighting(child, settings.combine)) for child in settings.checks
        )
        return f"{heading(settings)}\n\n{checks}"

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        settings = self._settings_of(check)
        marked = marked_in_order(settings.checks, context, question, [value] * len(settings.checks))
        counted: Counted = [
            (child, {key: replace(o, name=child.name) for key, o in outcomes.items()})
            for child, outcomes in marked
        ]
        return combine(settings, counted, context.questions)

    def ordered(self, settings: Mapping[str, Any]) -> bool:
        return True

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return marked_by(self._settings_of(check).checks, question)

    def questions_read(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return read_by(self._settings_of(check).checks, question)

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        # Only a written list needs a check: a method file may fill one from an optional input.
        if not self._settings_of(check).checks and not check.scope:
            yield Problem("Add at least one check.", field=("checks",), incomplete=True)


def heading(settings: CombinationSettings, *notes: str) -> str:
    """How a description of combined checks opens: ``**Each one counts, in any order**``."""
    return f"**{', '.join([HEADINGS[settings.combine], *notes])}**"


def _taken_off(check: Check) -> str:
    return f"Marks taken off: {check.name}." if check.name else "Marks taken off."


def _weighting(check: Check, combine: Combine) -> tuple[str, ...]:
    """What a check's weight, when it isn't 1, makes it worth: under ``any`` the share of the
    marks it can give, and under ``total`` and ``proportional`` the weight itself."""
    if check.weight == 1 or combine in ("all", "none"):
        return ()
    if combine == "any":
        return (f"{check.weight:.0%} of the marks",)
    return (f"weight {format_number(check.weight)}",)


def _result_of(outcomes: dict[QuestionId, Outcome]) -> CheckResult:
    """A check's result over every question it marks."""
    values = list(outcomes.values())
    return {
        "correct": bool(values) and all(outcome.correct for outcome in values),
        "proportion": min((outcome.proportion for outcome in values), default=0.0),
        "feedback": "\n".join(outcome.feedback for outcome in values if outcome.feedback),
        "marked": all(outcome.marked for outcome in values),
        "skipped": bool(values) and all(outcome.skipped for outcome in values),
    }
