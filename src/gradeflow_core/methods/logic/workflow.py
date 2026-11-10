from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator

from ...common.hints import hint
from ...common.paths import Location
from ...common.problems import Problem
from ...common.text import as_sentence, format_number, markdown_list, plain_markdown
from ...questions.base import Question
from ...questions.question_set import QuestionSet
from ...questions.types import ALL_TYPES, QuestionId
from ..example import examples_beside
from ..input_sources import StepOutputSource, input_source
from ..method import Method
from ..outcome import (
    CORRECT,
    FULL_MARKS_PROPORTION,
    Effect,
    MethodResult,
    Outcome,
    Skipped,
    clamped,
    effect_of,
)
from ..position import Position
from ..settings import Settings, specialized
from ..step import AnyStep, Check, Context, Marking, Step
from ..templates import names_read, names_read_in

MarksBy = Literal["all_correct", "any_correct", "none_correct", "total", "proportional"]
ADDITIVE_MARKS_BY: frozenset[MarksBy] = frozenset({"total", "proportional"})
"""The ways of giving marks that add the parts' weighted marks up."""
MARKS_BY_LABELS = {
    "all_correct": "All must be correct",
    "any_correct": "Any one must be correct",
    "none_correct": "None may be correct",
    "total": "Total",
    "proportional": "Proportional",
}
NestedStep = Annotated[AnyStep, Field(title="Step", json_schema_extra=hint(kind="step"))]
NestedCheck = Annotated[Marking, Field(title="Check", json_schema_extra=hint(kind="step"))]
MARKS_BY_FIELD: MarksBy = Field(
    default="all_correct",
    title="How marks are given",
    json_schema_extra=hint(labels=MARKS_BY_LABELS),
)


class WorkflowSettings(Settings):
    output: Literal["marks", "step_output"] = Field(
        default="marks",
        title="Output",
        json_schema_extra=hint(labels={"marks": "Marks", "step_output": "A step's output"}),
    )
    marks_by: MarksBy = Field(
        default="all_correct",
        title="How marks are given",
        json_schema_extra=hint(labels=MARKS_BY_LABELS, applies_when={"output": ["marks"]}),
    )
    output_step: str | None = Field(
        default=None,
        title="Step",
        description="The step whose output this gives.",
        json_schema_extra=hint(kind="output_step", applies_when={"output": ["step_output"]}),
    )
    steps: list[NestedStep] = Field(
        default_factory=list,
        title="Steps",
        json_schema_extra=hint(item_label="step"),
    )

    @field_validator("output_step")
    @classmethod
    def _validate_output_step(cls, output_step: str | None) -> str | None:
        if output_step is not None and not isinstance(input_source(output_step), StepOutputSource):
            raise ValueError("Name a step, such as {{ steps.name.output }}.")
        return output_step


DESCRIPTION_HEADINGS = {
    "all_correct": "All of these",
    "any_correct": "Any one of these",
    "none_correct": "None of these",
    "total": "Total of these",
    "proportional": "Each one counts",
}
"""How a workflow's description opens."""

type CheckOutcomes = list[tuple[Check, dict[QuestionId, Outcome]]]


def run_steps(
    steps: Sequence[Step], contexts: Sequence[Context], question: QuestionId | None
) -> tuple[list[CheckOutcomes], list[Context]]:
    """Each student's checks and their outcomes, and each student's context after the steps."""
    contexts = list(contexts)
    check_outcomes: list[CheckOutcomes] = [[] for _ in contexts]
    for step in steps:
        results_by_student = step.run(contexts, question)
        contexts = [context for context, _ in results_by_student]
        for student_outcomes, (_, outcomes) in zip(check_outcomes, results_by_student, strict=True):
            if isinstance(step, Check) and outcomes is not None:
                student_outcomes.append((step, outcomes))
    return check_outcomes, contexts


def combine(
    marks_by: MarksBy, check_outcomes: CheckOutcomes, questions: QuestionSet
) -> dict[QuestionId, Outcome]:
    """One outcome for each question the checks mark, combined as ``marks_by`` says. A question
    whose checks were all skipped is skipped."""
    scoring_checks = [(check, outcomes) for check, outcomes in check_outcomes if check.has_weight]
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
        if marks_by == "any_correct"
        else (None, _combined_by_question(marks_by, unskipped, questions))
    )
    return {
        key: replace(
            outcome,
            children=_children(
                marks_by,
                check_outcomes,
                key,
                best_check if outcome.proportion > 0 else None,
                _max_marks(questions, key),
            ),
        )
        for key, outcome in combined.items()
    }


def _combined_by_question(
    marks_by: MarksBy, unskipped: CheckOutcomes, questions: QuestionSet
) -> dict[QuestionId, Outcome]:
    keys = dict.fromkeys(key for _, outcomes in unskipped for key in outcomes)
    combined: dict[QuestionId, Outcome] = {}
    for key in keys:
        checks_with_parts = [
            (check, outcomes[key]) for check, outcomes in unskipped if key in outcomes
        ]
        parts = tuple(outcome for _, outcome in checks_with_parts)
        weighted_parts = [(check.weight, outcome) for check, outcome in checks_with_parts]
        proportion = _proportion(marks_by, weighted_parts, _max_marks(questions, key))
        if marks_by == "none_correct":
            error = next(filter(None, (part.first_error() for part in parts)), None)
            proportion = 0.0 if error else proportion
            feedback = _not_allowed_feedback(checks_with_parts, error)
        else:
            feedback = _joined_feedback(
                _explaining_parts(marks_by, checks_with_parts, proportion >= FULL_MARKS_PROPORTION)
            )
        combined[key] = Outcome(
            proportion, feedback, marked=all(part.marked for part in parts), parts=parts
        )
    return combined


def _children(
    marks_by: MarksBy,
    check_outcomes: CheckOutcomes,
    key: QuestionId,
    best_check: Check | None,
    max_marks: float,
) -> tuple[Outcome, ...]:
    """The checks that ran, each with its effect."""
    ran_parts = [
        (check, part)
        for check, outcomes in check_outcomes
        if (part := _outcome_for_question(check, outcomes, key)) is not None and not part.skipped
    ]
    scoring_parts = [(check, part) for check, part in ran_parts if check.has_weight]
    best_index = next(
        (index for index, (check, _) in enumerate(scoring_parts) if check is best_check),
        None,
    )
    scoring_parts_with_effects = iter(
        parts_with_effects(
            marks_by, [(check.weight, part) for check, part in scoring_parts], best_index, max_marks
        )
    )
    return tuple(
        next(scoring_parts_with_effects) if check.has_weight else _with_condition_effect(part)
        for check, part in ran_parts
    )


def proportion_and_parts(
    marks_by: MarksBy, weighted_parts: Sequence[tuple[float, Outcome]], max_marks: float
) -> tuple[float, tuple[Outcome, ...]]:
    """The combined proportion of ``(weight, outcome)`` parts, and the parts with their effects."""
    proportion = _proportion(marks_by, weighted_parts, max_marks)
    best_index = (
        max(
            range(len(weighted_parts)),
            key=lambda index: weighted_parts[index][0] * weighted_parts[index][1].proportion,
        )
        if marks_by == "any_correct" and proportion > 0
        else None
    )
    return proportion, parts_with_effects(marks_by, weighted_parts, best_index, max_marks)


def _proportion(
    marks_by: MarksBy, weighted_parts: Sequence[tuple[float, Outcome]], max_marks: float
) -> float:
    """The proportion ``(weight, outcome)`` parts of one answer give together."""
    weighted_proportions = [(weight, part.proportion) for weight, part in weighted_parts]
    out_of = marks_out_of(marks_by, [weight for weight, _ in weighted_parts], max_marks)
    return clamped(combined_proportion(marks_by, weighted_proportions, out_of))


def parts_with_effects(
    marks_by: MarksBy,
    weighted_parts: Sequence[tuple[float, Outcome]],
    best_index: int | None,
    max_marks: float,
) -> tuple[Outcome, ...]:
    """Combined ``(weight, outcome)`` parts, each with what it did to the marks and, under "Total"
    and "Proportional", the part of the combined proportion it gave. Only the part at
    ``best_index``, if there is one, counts."""
    out_of = marks_out_of(marks_by, [weight for weight, _ in weighted_parts], max_marks)

    def with_effect(index: int, weight: float, part: Outcome) -> Outcome:
        if best_index is not None and index != best_index:
            return replace(part, effect="uncounted")
        contribution = weight * part.proportion / out_of if marks_by in ADDITIVE_MARKS_BY else None
        return replace(part, effect=_effect(marks_by, weight, part), contribution=contribution)

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
    """A check's outcome for ``key``, or a no-weight check's outcome for the one question it
    reads."""
    if key in outcomes:
        return outcomes[key]
    if check.has_weight or len(outcomes) != 1:
        return None
    [(other_question, outcome)] = outcomes.items()
    return replace(outcome, labels=(*outcome.labels, other_question))


def _effect(marks_by: MarksBy, weight: float, part: Outcome) -> Effect:
    """What ``part`` did to the marks. Under "None may be correct", or with a negative weight, a
    check that finds what it looks for is a penalty."""
    if not part.marked or not (
        marks_by == "none_correct" or (marks_by in ADDITIVE_MARKS_BY and weight <= 0)
    ):
        return effect_of(part)
    return "penalty" if part.proportion > 0 and weight != 0 else "uncounted"


def _explaining_parts(
    marks_by: MarksBy, checks_with_parts: list[tuple[Check, Outcome]], correct: bool
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
        is_additive = marks_by in ADDITIVE_MARKS_BY
        weight = check.weight if is_additive else 1
        match marks_by:
            case "all_correct":
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
    """Feedback under "None may be correct": why parts weren't marked, and what was found, or
    "Correct."."""
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


def marks_out_of(marks_by: MarksBy, weights: Sequence[float], max_marks: float) -> float:
    """What additive marks are out of: ``max_marks`` for a total, or the positive ``weights`` of
    every check counted."""
    return (max_marks if marks_by == "total" else sum(w for w in weights if w > 0)) or 1


def combined_proportion(
    marks_by: MarksBy, weighted_proportions: Sequence[tuple[float, float]], out_of: float
) -> float:
    """The proportion of ``(weight, proportion)`` parts of one answer combined, before clamping.
    Additive marks are ``out_of`` the number `marks_out_of` gives."""
    proportions = [proportion for _, proportion in weighted_proportions]
    earned = sum(weight * proportion for weight, proportion in weighted_proportions)
    match marks_by:
        case "all_correct":
            return min(proportions)
        case "none_correct":
            return 1 - max(proportions)
        case "any_correct":
            return max(weight * proportion for weight, proportion in weighted_proportions)
        case "total" | "proportional":
            return earned / out_of


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


def _problems_of_steps(
    steps: Sequence[Step], step_ids_read_by_output: frozenset[str] = frozenset()
) -> Iterator[Problem]:
    for index, inner in enumerate(steps):
        step_ids_read_later = names_read_in([step.dump() for step in steps[index + 1 :]], "steps")
        yield from inner.problems_in_list(step_ids_read_later | step_ids_read_by_output, index)


class Workflow(Method[WorkflowSettings]):
    name = "gradeflow/workflow"
    version = 1
    label = "Workflow"
    summary = "Several steps combined into one mark."
    guide = (
        "**Use it when:** One check isn't enough, such as an answer with reasoning and a final "
        "number, or an answer that must name a key term and stay within a word limit.\n\n"
        "**How it marks:** Steps run in order. **How marks are given** sets how the checks count: "
        "**All must be correct** and **None may be correct** give full marks or none, **Any one "
        "must be correct** counts the check that gives the most marks, **Total** adds up the "
        "correct checks by their weights, as marks of the question, and **Proportional** gives "
        "each its share of the marks by its weight. A transform gives no marks: it finds part of "
        "the answer for a later step to read as its **Input**.\n\n"
        "**Good to know:** A weight below 1, such as `0.5`, makes a check worth part of the marks. "
        "A weight of `0` lets later steps read a check without it giving marks, and a negative "
        "weight is a penalty. With **Any one must be correct**, list the correct answer first, "
        "then partly correct answers with lower weights.\n\n"
        "**Related:** To forbid something, add a Not Allowed check. To mark several questions "
        "together, make a question group."
    )
    level = "advanced"
    question_types = ALL_TYPES
    marks_question_group = True
    settings = WorkflowSettings
    examples = examples_beside(__file__)

    def gives_marks(self, settings: Mapping[str, Any]) -> bool:
        return bool(settings.get("output", "marks") == "marks")

    def settings_model_at(self, position: Position) -> type[Settings]:
        """Outside a workflow, nothing can read a step's output."""
        if position.in_workflow:
            return self.settings
        return specialized(
            self.settings,
            without=("output", "output_step"),
            marks_by={"json_schema_extra": hint(labels=MARKS_BY_LABELS)},
        )

    def inner_position(self, position: Position, location: Location) -> Position:
        return replace(super().inner_position(position, location), in_workflow=True)

    def reads_earlier_results(self, settings: Mapping[str, Any]) -> bool:
        return True

    def describe(self, step: Step) -> str:
        settings = self._settings_of(step)
        if settings.output == "step_output":
            return "\n".join(inner.describe() for inner in settings.steps)
        lines = markdown_list(
            inner.described_in_list(weight_notes(inner.marks_weight, settings.marks_by))
            for inner in settings.steps
        )
        return f"{description_heading(settings.marks_by)}\n{lines}"

    def run(
        self, step: Step, contexts: Sequence[Context], question: QuestionId | None
    ) -> list[MethodResult]:
        settings = self._settings_of(step)
        check_outcomes_by_student, final_contexts = run_steps(settings.steps, contexts, question)
        if settings.output == "step_output":
            assert settings.output_step is not None  # problems refuse a workflow without one
            [step_id] = names_read(settings.output_step, "steps")
            return [
                result.transform_result
                if (result := context.earlier_results.get(step_id))
                else Skipped()
                for context in final_contexts
            ]
        return [
            _combined_for_student(settings.marks_by, student_check_outcomes, context)
            for context, student_check_outcomes in zip(
                contexts, check_outcomes_by_student, strict=True
            )
        ]

    def output_question(
        self,
        settings: Mapping[str, Any],
        source_question: Question,
        methods: Mapping[str, Method[Any]],
    ) -> Question | None:
        if self.gives_marks(settings):
            return None
        output_questions: dict[str, Question] = {}
        steps = settings.get("steps")
        for entry in steps if isinstance(steps, list) else ():
            method = methods.get(str(entry.get("uses"))) if isinstance(entry, Mapping) else None
            if method is not None and isinstance(entry.get("id"), str):
                input_question = input_source(entry.get("input")).input_question(
                    output_questions, source_question
                )
                step_output_question = method.output_question(
                    entry.get("with") or {}, input_question, methods
                )
                if step_output_question is not None:
                    output_questions[entry["id"]] = step_output_question
        return input_source(settings.get("output_step")).input_question(
            output_questions, source_question
        )

    def problems(self, step: Step, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(step)
        steps = settings.steps
        # Only a written list needs a step: a method file may fill one from an optional setting.
        if not steps and not step._scope:  # noqa: SLF001
            yield Problem("Add at least one step.", field=("steps",), is_incomplete=True)
        if settings.output == "step_output":
            yield from self._output_step_problems(settings)
            return
        if steps and not any(inner.has_weight for inner in steps):
            yield Problem("Add a step that gives marks.", field=("steps",), is_incomplete=True)
        yield from _problems_of_steps(steps)
        yield from ignored_weight_problems(
            [inner.marks_weight for inner in steps], settings.marks_by, "steps"
        )

    def _output_step_problems(self, settings: WorkflowSettings) -> Iterator[Problem]:
        step_ids_read_by_output = frozenset[str]()
        if settings.output_step is None:
            yield Problem(
                "Choose the step whose output it gives.", field=("output_step",), is_incomplete=True
            )
        else:
            step_ids_read_by_output = names_read(settings.output_step, "steps")
            for missing in step_ids_read_by_output - {inner.id for inner in settings.steps}:
                yield Problem(f"No step here is called {missing}.", field=("output_step",))
        yield from _problems_of_steps(settings.steps, step_ids_read_by_output)


def _combined_for_student(
    marks_by: MarksBy, check_outcomes: CheckOutcomes, context: Context
) -> dict[QuestionId, Outcome]:
    named_check_outcomes: CheckOutcomes = [
        (check, {key: replace(outcome, name=check.name) for key, outcome in outcomes.items()})
        for check, outcomes in check_outcomes
    ]
    combined = combine(marks_by, named_check_outcomes, context.questions)
    # A check reading a transform keeps its row, which says what it read.
    if len(check_outcomes) > 1 or any(check.input for check, _ in check_outcomes):
        return combined
    return {key: _collapsed_to_only_child(outcome) for key, outcome in combined.items()}


def _collapsed_to_only_child(outcome: Outcome) -> Outcome:
    """A workflow of one check, replaced by its only child."""
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


def description_heading(marks_by: MarksBy, *notes: str) -> str:
    """How a description of combined checks opens: ``Each one counts, in any order:``."""
    return f"{', '.join([DESCRIPTION_HEADINGS[marks_by], *notes])}:"


def _penalty_feedback(check: Check) -> str:
    return f"Penalty: {check.name}." if check.name else "Penalty."


def weight_notes(weight: float, marks_by: MarksBy) -> tuple[str, ...]:
    """What a weight, when it isn't 1, makes a check worth: under ``any_correct`` the share of the
    marks it can give, and under ``total`` and ``proportional`` the weight itself."""
    if weight == 1:
        return ()
    if marks_by in ADDITIVE_MARKS_BY:
        return (f"weight {format_number(weight)}",)
    return (f"{weight:.0%} of the marks",) if marks_by == "any_correct" else ()


def ignored_weight_problems(
    weights: Sequence[float], marks_by: MarksBy, field: str
) -> Iterator[Problem]:
    """A problem at each weight other than 0 and 1, when ``marks_by`` ignores weights."""
    if marks_by in ("all_correct", "none_correct"):
        for index, weight in enumerate(weights):
            if weight not in (0, 1):
                yield Problem(
                    f"“{MARKS_BY_LABELS[marks_by]}” doesn't use weights.",
                    field=(field, index, "weight"),
                )
