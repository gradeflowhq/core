from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Annotated, Any, Literal

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import as_sentence, format_number, markdown_list, plain_markdown
from ...questions.question_set import QuestionSet
from ...questions.types import ALL_TYPES, QuestionId
from ..check import Check, Context, Marking
from ..example import examples_beside
from ..method import Method
from ..outcome import (
    CORRECT,
    FULL_MARKS_PROPORTION,
    EarlierResult,
    Effect,
    Outcome,
    clamped,
    effect_of,
)
from ..position import Position
from ..settings import Settings

Combine = Literal["all", "any", "none", "total", "proportional"]
ADDITIVE_COMBINES: frozenset[Combine] = frozenset({"total", "proportional"})
"""The ways of combining that add the parts' weighted marks up."""
COMBINE_LABELS = {
    "all": "All must be correct",
    "any": "Any one must be correct",
    "none": "None may be correct",
    "total": "Total",
    "proportional": "Proportional",
}
NestedCheck = Annotated[Marking, Field(title="Check", json_schema_extra=hint(kind="check"))]


class CombinationSettings(Settings):
    combine: Combine = Field(
        default="all",
        title="How to combine",
        json_schema_extra=hint(labels=COMBINE_LABELS),
    )
    checks: list[NestedCheck] = Field(
        default_factory=list, title="Checks", description="The checks to combine."
    )


DESCRIPTION_HEADINGS = {
    "all": "All of these",
    "any": "Any one of these",
    "none": "None of these",
    "total": "Total of these",
    "proportional": "Each one counts",
}
"""How a combination's description opens: what the answer does, so it reads after "Not allowed:"."""

type CheckOutcomes = list[tuple[Check, dict[QuestionId, Outcome]]]


def outcomes_in_order(
    checks: Sequence[Check],
    context: Context,
    question: QuestionId | None,
    values: Sequence[int | None],
) -> CheckOutcomes:
    """Each check's outcomes, the check reading ``question`` at its value of ``values``; a check's
    condition sees the results of the earlier ones."""
    earlier: dict[str, EarlierResult] = {}
    check_outcomes: CheckOutcomes = []
    for check, value in zip(checks, values, strict=True):
        context_with_earlier = replace(
            context, earlier_results={**context.earlier_results, **earlier}
        )
        outcomes = check.mark(context_with_earlier, question, value)
        if check.id:
            earlier[check.id] = _earlier_result(outcomes)
        check_outcomes.append((check, outcomes))
    return check_outcomes


def combine(
    settings: CombinationSettings, check_outcomes: CheckOutcomes, questions: QuestionSet
) -> dict[QuestionId, Outcome]:
    """One outcome for each question the checks mark, combined as ``settings`` says. A question
    whose checks were all skipped is skipped."""
    scoring_checks = [
        (check, outcomes) for check, outcomes in check_outcomes if not check.read_only
    ]
    unskipped = [
        (check, unskipped_outcomes)
        for check, outcomes in scoring_checks
        if (
            unskipped_outcomes := {
                key: outcome for key, outcome in outcomes.items() if not outcome.skipped
            }
        )
    ]
    if not unskipped:
        keys = dict.fromkeys(key for _, outcomes in scoring_checks for key in outcomes)
        return {key: Outcome(0.0, skipped=True) for key in keys}
    best_check, combined = (
        _best_check(unskipped, questions)
        if settings.combine == "any"
        else (None, _combined_by_question(settings, unskipped, questions))
    )
    return {
        key: replace(
            outcome,
            children=_children(
                settings.combine,
                check_outcomes,
                key,
                best_check,
                outcome,
                _max_marks(questions, key),
            ),
        )
        for key, outcome in combined.items()
    }


def _combined_by_question(
    settings: CombinationSettings, unskipped: CheckOutcomes, questions: QuestionSet
) -> dict[QuestionId, Outcome]:
    keys = dict.fromkeys(key for _, outcomes in unskipped for key in outcomes)
    combined: dict[QuestionId, Outcome] = {}
    for key in keys:
        checks_with_parts = [
            (check, outcomes[key]) for check, outcomes in unskipped if key in outcomes
        ]
        parts = tuple(outcome for _, outcome in checks_with_parts)
        weighted_parts = [(check.weight, outcome) for check, outcome in checks_with_parts]
        proportion = _proportion(settings.combine, weighted_parts, _max_marks(questions, key))
        if settings.combine == "none":
            error = next(filter(None, (part.first_error() for part in parts)), None)
            proportion = 0.0 if error else proportion
            feedback = _not_allowed_feedback(checks_with_parts, error)
        else:
            feedback = _joined_feedback(
                _explaining_parts(
                    settings.combine, checks_with_parts, proportion >= FULL_MARKS_PROPORTION
                )
            )
        combined[key] = Outcome(
            proportion, feedback, marked=all(part.marked for part in parts), parts=parts
        )
    return combined


def _children(
    combine: Combine,
    check_outcomes: CheckOutcomes,
    key: QuestionId,
    best_check: Check | None,
    combined: Outcome,
    max_marks: float,
) -> tuple[Outcome, ...]:
    """The checks that ran: read-only ones with what they decided, the others counted."""
    ran_parts = [
        (check, part)
        for check, outcomes in check_outcomes
        if (part := _outcome_for_question(check, outcomes, key)) is not None and not part.skipped
    ]
    scoring_parts = [(check, part) for check, part in ran_parts if not check.read_only]
    best = (
        next(
            (index for index, (check, _) in enumerate(scoring_parts) if check is best_check),
            None,
        )
        if combined.proportion > 0
        else None
    )
    scoring_parts_with_effects = iter(
        parts_with_effects(
            combine, [(check.weight, part) for check, part in scoring_parts], best, max_marks
        )
    )
    return tuple(
        _with_condition_effect(part) if check.read_only else next(scoring_parts_with_effects)
        for check, part in ran_parts
    )


def proportion_and_parts(
    combine: Combine, weighted_parts: Sequence[tuple[float, Outcome]], max_marks: float
) -> tuple[float, tuple[Outcome, ...]]:
    """The proportion ``(weight, outcome)`` parts give together, and the parts, counted."""
    proportion = _proportion(combine, weighted_parts, max_marks)
    best = (
        max(
            range(len(weighted_parts)),
            key=lambda index: weighted_parts[index][0] * weighted_parts[index][1].proportion,
        )
        if combine == "any" and proportion > 0
        else None
    )
    return proportion, parts_with_effects(combine, weighted_parts, best, max_marks)


def _proportion(
    combine: Combine, weighted_parts: Sequence[tuple[float, Outcome]], max_marks: float
) -> float:
    """The proportion ``(weight, outcome)`` parts of one answer give together."""
    weighted_proportions = [(weight, part.proportion) for weight, part in weighted_parts]
    weights = [weight for weight, _ in weighted_parts]
    return clamped(combined_proportion(combine, weighted_proportions, weights, max_marks))


def parts_with_effects(
    combine: Combine,
    weighted_parts: Sequence[tuple[float, Outcome]],
    best: int | None,
    max_marks: float,
) -> tuple[Outcome, ...]:
    """Combined ``(weight, outcome)`` parts, each with what it did to the marks and, under "Total"
    and "Proportional", the part of the combined proportion it gave. Only the part at ``best``, if
    there is one, counts."""
    positive_weight_total = sum(weight for weight, _ in weighted_parts if weight > 0) or 1

    def with_effect(index: int, weight: float, part: Outcome) -> Outcome:
        if best is not None and index != best:
            return replace(part, effect="uncounted")
        match combine:
            case "total":
                contribution: float | None = weight * part.proportion / (max_marks or 1)
            case "proportional":
                contribution = weight * part.proportion / positive_weight_total
            case _:
                contribution = None
        return replace(part, effect=_effect(combine, weight, part), contribution=contribution)

    return tuple(
        with_effect(index, weight, part) for index, (weight, part) in enumerate(weighted_parts)
    )


def _with_condition_effect(part: Outcome) -> Outcome:
    if not part.marked:
        return replace(part, effect="unmarked")
    return replace(part, effect="met" if part.correct else "not_met")


def _outcome_for_question(
    check: Check, outcomes: dict[QuestionId, Outcome], key: QuestionId
) -> Outcome | None:
    """A check's outcome for ``key``; a read-only check's for the one other question it reads,
    titled with that question."""
    if key in outcomes:
        return outcomes[key]
    if not check.read_only or len(outcomes) != 1:
        return None
    [(other_question, outcome)] = outcomes.items()
    return replace(outcome, labels=(*outcome.labels, other_question))


def _effect(combine: Combine, weight: float, part: Outcome) -> Effect:
    """What ``part`` did to the marks. Under "None", or with a negative weight, a check that finds
    what it looks for is a penalty."""
    if not part.marked or not (combine == "none" or (combine in ADDITIVE_COMBINES and weight <= 0)):
        return effect_of(part)
    return "penalty" if part.proportion > 0 and weight != 0 else "uncounted"


def _explaining_parts(
    combine: Combine, checks_with_parts: list[tuple[Check, Outcome]], correct: bool
) -> tuple[Outcome, ...]:
    """The parts that explain the combined proportion: what counted for it, and, unless the whole
    is correct, what counted against it. Only "Total" and "Proportional" keep what counted for an
    incorrect whole, as each part earned its share. A penalty says only that it applies, unless
    its feedback is its own."""
    explaining_parts = []
    for check, part in checks_with_parts:
        if not part.marked:  # says why the whole is left for a person
            explaining_parts.append(part)
            continue
        is_additive = combine in ADDITIVE_COMBINES
        weight = check.weight if is_additive else 1
        match combine:
            case "all":
                gives_marks, is_penalty = part.correct, not part.correct
            case _:
                gives_marks = weight * part.proportion > 0
                is_penalty = weight * part.proportion < 0 or (weight > 0 and not part.correct)
        if (gives_marks and (correct or is_additive)) or (is_penalty and not correct):
            if weight < 0 and not check.feedback:
                part = replace(part, feedback=_penalty_feedback(check), name=None)
            explaining_parts.append(part)
    return tuple(explaining_parts)


def _not_allowed_feedback(checks_with_parts: list[tuple[Check, Outcome]], error: str | None) -> str:
    """Under "None may be correct": why parts are left for a person, and the ``error`` that kept a
    part from checking the answer, which gets no marks then; or "Not allowed." with what each check
    that found the answer checks, unless its feedback is its own; else "Correct."."""
    not_allowed_reasons = [
        part.feedback if check.feedback else as_sentence(plain_markdown(check.describe()))
        for check, part in checks_with_parts
        if part.marked and part.proportion > 0
    ]
    unmarked_feedback = _joined_feedback(
        tuple(part for _, part in checks_with_parts if not part.marked)
    )
    not_allowed_line = (
        as_sentence(f"Couldn't check the answer: {error}")
        if error
        else (f"Not allowed. {' '.join(not_allowed_reasons)}" if not_allowed_reasons else "")
    )
    return "\n".join(line for line in (unmarked_feedback, not_allowed_line) if line) or CORRECT


def combined_proportion(
    combine: Combine,
    weighted_proportions: Sequence[tuple[float, float]],
    weights: Sequence[float],
    max_marks: float,
) -> float:
    """The proportion of ``(weight, proportion)`` parts of one answer combined, before clamping.
    A total is marks of a question of ``max_marks``; a proportional share is out of the positive
    ``weights`` of every check counted."""
    proportions = [proportion for _, proportion in weighted_proportions]
    earned = sum(weight * proportion for weight, proportion in weighted_proportions)
    match combine:
        case "all":
            return min(proportions)
        case "none":
            return 1 - max(proportions)
        case "any":
            return max(weight * proportion for weight, proportion in weighted_proportions)
        case "total":
            return earned / (max_marks or 1)
        case "proportional":
            return earned / (sum(weight for weight in weights if weight > 0) or 1)


def _best_check(
    check_outcomes: CheckOutcomes, questions: QuestionSet
) -> tuple[Check, dict[QuestionId, Outcome]]:
    """The check giving the most marks over the questions it marks, the first on a tie, and its
    outcomes. Questions only other checks mark get no marks. An answer stays unmarked while a check
    that couldn't mark it could give more."""

    def total(item: tuple[Check, dict[QuestionId, Outcome]]) -> float:
        check, outcomes = item
        return sum(
            check.weight * outcome.proportion * _max_marks(questions, key)
            for key, outcome in outcomes.items()
        )

    best_check, best_outcomes = max(check_outcomes, key=total)

    def best_feedback(outcome: Outcome) -> str:
        # A correct check worth part of the marks, such as "Sign error", says what earned them
        # better than its method's own "Correct.", unless its author wrote its feedback. Otherwise
        # the best check speaks for the answer, without its name.
        if outcome.proportion == 1 and 0 < best_check.weight < 1 and not best_check.feedback:
            return f"{outcome.name or 'Partly correct'} ({best_check.weight:.0%} of the marks)."
        return outcome.feedback

    def best_outcome(key: QuestionId, outcome: Outcome) -> Outcome:
        proportion = clamped(best_check.weight * outcome.proportion)
        unmarked_worth_more = tuple(
            part
            for check, outcomes in check_outcomes
            if check is not best_check
            and (part := outcomes.get(key)) is not None
            and not part.marked
            and check.weight > proportion
        )
        feedback_lines = (_joined_feedback(unmarked_worth_more), best_feedback(outcome))
        return Outcome(
            proportion,
            "\n".join(line for line in feedback_lines if line),
            marked=outcome.marked and not unmarked_worth_more,
            parts=tuple(outcomes[key] for _, outcomes in check_outcomes if key in outcomes),
        )

    return best_check, {key: best_outcome(key, outcome) for key, outcome in best_outcomes.items()}


def _max_marks(questions: QuestionSet, key: QuestionId) -> float:
    question = questions.get(key)
    return question.max_marks if question else 1.0


def _joined_feedback(parts: tuple[Outcome, ...]) -> str:
    """The parts' feedback, each line once. A bare "Correct." adds nothing to other lines."""
    lines = list(
        dict.fromkeys(_with_check_name(part.name, part.feedback) for part in parts if part.feedback)
    )
    lines_besides_correct = [line for line in lines if line != CORRECT]
    return "\n".join(lines_besides_correct or lines)


def _with_check_name(name: str | None, feedback: str) -> str:
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
        "part of the marks, and a negative weight is a penalty.\n\n"
        "**Good to know:** With **Any one must be correct**, list the correct answer first, then "
        "partly correct answers with lower weights.\n\n"
        "**Related:** To forbid something, add a Not Allowed check. To mark several questions "
        "together, make a question group."
    )
    level = "advanced"
    question_types = ALL_TYPES
    marks_question_group = True
    settings = CombinationSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        checks = markdown_list(
            inner_check.described_with_heading(weight_notes(inner_check.weight, settings.combine))
            for inner_check in settings.checks
        )
        return f"{description_heading(settings)}\n{checks}"

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        settings = self._settings_of(check)
        checks = settings.checks
        check_outcomes: CheckOutcomes = [
            (
                inner_check,
                {key: replace(outcome, name=inner_check.name) for key, outcome in outcomes.items()},
            )
            for inner_check, outcomes in outcomes_in_order(
                checks, context, question, [value] * len(checks)
            )
        ]
        combined = combine(settings, check_outcomes, context.questions)
        if len(checks) > 1:
            return combined
        return {key: _collapsed_to_only_child(outcome) for key, outcome in combined.items()}

    def reads_earlier_results(self, settings: Mapping[str, Any]) -> bool:
        return True

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        # Only a written list needs a check: a method file may fill one from an optional input.
        if not settings.checks and not check._from_method_file:  # noqa: SLF001
            yield Problem("Add at least one check.", field=("checks",), is_incomplete=True)
        yield from ignored_weight_problems(
            [inner.weight for inner in settings.checks], settings.combine, "checks"
        )


def _collapsed_to_only_child(outcome: Outcome) -> Outcome:
    """A combination of one check, replaced by its only child."""
    if len(outcome.children) != 1:
        return outcome
    [only_child] = outcome.children
    return replace(
        outcome,
        title=only_child.title,
        labels=only_child.labels,
        details=only_child.details,
        children=only_child.children,
    )


def description_heading(settings: CombinationSettings, *notes: str) -> str:
    """How a description of combined checks opens: ``Each one counts, in any order:``."""
    return f"{', '.join([DESCRIPTION_HEADINGS[settings.combine], *notes])}:"


def _penalty_feedback(check: Check) -> str:
    return f"Penalty: {check.name}." if check.name else "Penalty."


def weight_notes(weight: float, combine: Combine) -> tuple[str, ...]:
    """What a weight, when it isn't 1, makes a check worth: under ``any`` the share of the marks
    it can give, and under ``total`` and ``proportional`` the weight itself."""
    if weight == 1:
        return ()
    if combine in ADDITIVE_COMBINES:
        return (f"weight {format_number(weight)}",)
    return (f"{weight:.0%} of the marks",) if combine == "any" else ()


def ignored_weight_problems(
    weights: Sequence[float], combine: Combine, field: str
) -> Iterator[Problem]:
    """A problem at each weight but 1, as ``all`` and ``none`` don't use weights."""
    if combine in ("all", "none"):
        for index, weight in enumerate(weights):
            if weight != 1:
                yield Problem(
                    f"“{COMBINE_LABELS[combine]}” doesn't use weights.",
                    field=(field, index, "weight"),
                )


def _earlier_result(outcomes: dict[QuestionId, Outcome]) -> EarlierResult:
    """A check's result over every question it marks."""
    return {
        "correct": bool(outcomes) and all(outcome.correct for outcome in outcomes.values()),
        "proportion": min((outcome.proportion for outcome in outcomes.values()), default=0.0),
        "feedback": "\n".join(
            outcome.feedback for outcome in outcomes.values() if outcome.feedback
        ),
        "marked": all(outcome.marked for outcome in outcomes.values()),
        "skipped": bool(outcomes) and all(outcome.skipped for outcome in outcomes.values()),
    }
