"""Running a step for each student: reading its input, testing its condition, and making what its
method gives into outcomes or an answer for later steps."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..questions.base import Question
from ..questions.types import QuestionId, Unreadable
from .context import Context, StepInput
from .input_sources import AnswerSource, AnswerValueSource, OtherAnswerSource, StepOutputSource
from .outcome import (
    CheckOutput,
    Detail,
    Outcome,
    QuestionOutcomes,
    Skipped,
    StepResult,
    outcome_of,
    unanswered_outcome,
)
from .places import output_question
from .position import Position
from .settings import Settings
from .step import OutcomeFeedback, Step, method_and_settings
from .templates import TemplateError, condition_expression, evaluate, names_read, render


@dataclass(frozen=True)
class Blank:
    """A blank answer, which a method doesn't mark unless it ``marks_blank_answers``."""


type NotRunReason = Skipped | Unreadable | Blank
"""Why a step doesn't run for a student: it is skipped, or its answer is unreadable or blank."""
type CheckOutcomes = list[tuple[Step, QuestionOutcomes]]


def run_steps(
    steps: Sequence[Step], contexts: Sequence[Context], question_id: QuestionId | None
) -> tuple[list[CheckOutcomes], list[Context]]:
    """Each student's checks and their outcomes, and each student's context after the steps."""
    contexts = list(contexts)
    check_outcomes_by_student: list[CheckOutcomes] = [[] for _ in contexts]
    for step in steps:
        if step.resolution.gives_marks:
            outcomes_by_student = mark(step, contexts, question_id)
            results = [_check_result(outcomes) for outcomes in outcomes_by_student]
            for student_check_outcomes, outcomes in zip(
                check_outcomes_by_student, outcomes_by_student, strict=True
            ):
                student_check_outcomes.append((step, outcomes))
        else:
            results = transform(step, contexts, question_id)
        if step.id:
            contexts = [
                replace(context, earlier_results={**context.earlier_results, step.id: result})
                for context, result in zip(contexts, results, strict=True)
            ]
    return check_outcomes_by_student, contexts


def marked_question_id(step: Step, question_id: QuestionId | None) -> QuestionId | None:
    """The key of the answer ``step`` reads and marks, at ``question_id`` unless it names its
    own."""
    question_id = step.question or question_id
    if question_id is None and isinstance(step.source, OtherAnswerSource):
        return step.source.question_id
    return question_id


def mark(
    check: Step, contexts: Sequence[Context], question_id: QuestionId | None = None
) -> list[QuestionOutcomes]:
    """What each student in ``contexts`` gets for each question, reading ``question_id`` unless
    the check names its own."""
    method, _ = method_and_settings(check)
    outcomes_by_student: list[QuestionOutcomes] = []
    for context, result in _results(check, contexts, question_id, method.mark_students):
        if isinstance(result, Skipped | Unreadable | Blank):
            marked_id = marked_question_id(check, question_id)
            question_ids = (
                {marked_id}
                if marked_id
                else check.questions_marked(question_id) or check.questions_read(question_id)
            )
            outcomes = dict.fromkeys(question_ids, _not_run_result(result))
        else:
            outcomes = {
                result_question_id: _with_feedback_and_details(
                    check, outcome, context, result_question_id
                )
                for result_question_id, outcome in result.items()
            }
        outcomes_by_student.append(
            {
                result_question_id: _titled(check, outcome, result_question_id)
                if isinstance(outcome, Outcome)
                else outcome
                for result_question_id, outcome in outcomes.items()
            }
        )
    return outcomes_by_student


def transform(
    step: Step, contexts: Sequence[Context], question_id: QuestionId | None
) -> list[StepResult]:
    """What a transform gives each student in ``contexts``, reading ``question_id`` unless it
    names its own."""
    method, _ = method_and_settings(step)
    results = _results(step, contexts, question_id, method.transform_students)
    step_input = next(
        (context.input for context, result in results if not isinstance(result, Skipped)), None
    )
    question = _output_question(step, contexts[0], step_input) if step_input else None
    return [
        StepResult(None, skipped=True)
        if isinstance(result, Skipped)
        else StepResult(None if isinstance(result, Blank) else result, question=question)
        for _, result in results
    ]


def _results[R](
    step: Step,
    contexts: Sequence[Context],
    question_id: QuestionId | None,
    run: Callable[[Settings, Sequence[Context], QuestionId | None], Sequence[R]],
) -> list[tuple[Context, R | NotRunReason]]:
    """Each student's context reading the step's input, with what ``run`` gives of it, or why
    the step doesn't run."""
    method, settings = method_and_settings(step)
    input_contexts = [
        _input_context(step, context, question_id, marks_blank_answers=method.marks_blank_answers)
        for context in contexts
    ]
    contexts_to_run = [context for context, reason in input_contexts if reason is None]
    results = iter(
        run(settings, contexts_to_run, marked_question_id(step, question_id))
        if contexts_to_run
        else ()
    )
    return [
        (context, next(results) if reason is None else reason) for context, reason in input_contexts
    ]


def _output_question(step: Step, context: Context, step_input: StepInput) -> Question | None:
    """The question that reads what a transform reading ``step_input`` gives."""
    position = Position(
        context.questions, step_input.question_id, input_question=step_input.question
    )
    return output_question(step, position)


def _input_context(
    step: Step, context: Context, question_id: QuestionId | None, *, marks_blank_answers: bool
) -> tuple[Context, NotRunReason | None]:
    """The student's context reading the step's input, and why the step doesn't run for them,
    if it doesn't."""
    step_input = _input(step, context, question_id)
    if isinstance(step_input, Skipped):
        return context, step_input
    not_run_reason: NotRunReason | None = _condition_not_run_reason(step, context)
    if not_run_reason is None and step_input is not None:
        if isinstance(step_input.answer, Unreadable):
            not_run_reason = step_input.answer
        elif step_input.answer is None and not marks_blank_answers:
            not_run_reason = Blank()
    return replace(context, input=step_input), not_run_reason


def _input(
    step: Step, context: Context, question_id: QuestionId | None
) -> StepInput | Skipped | None:
    """What the step reads for one student: nothing when it reads no one answer, and skipped when
    the step it reads didn't run."""
    marked_id = marked_question_id(step, question_id)
    if marked_id is None:
        return None
    match step.source:
        case AnswerSource():
            # A step reads what the step holding it reads, unless it names another question.
            enclosing_input = context.input
            if enclosing_input is not None and enclosing_input.question_id == marked_id:
                return enclosing_input
            return context.answer_to(marked_id)
        case AnswerValueSource(value_number=value_number):
            return context.value(marked_id, value_number)
        case OtherAnswerSource(question_id=other_question_id):
            return context.input_from(
                marked_id,
                context.answers.get(other_question_id),
                context.questions[other_question_id],
            )
        case StepOutputSource(step_id=step_id):
            result = context.earlier_results.get(step_id)
            if (
                result is None
                or result.question is None
                or isinstance(answer := result.transform_result, Skipped)
            ):
                return Skipped()
            return context.input_from(marked_id, answer, result.question)


def _condition_not_run_reason(step: Step, context: Context) -> Skipped | Unreadable | None:
    if step.if_ is None:
        return None
    expression = condition_expression(step.if_)
    for step_id in sorted(names_read(step.if_, "steps")):
        if step_id not in context.earlier_results:
            raise TemplateError(f"There's no earlier step {step_id}.")
        not_run_reason = context.earlier_results[step_id].condition_not_run_reason
        if not_run_reason is not None:
            return not_run_reason
    if not evaluate(expression, {**step.resolution.scope, "steps": context.earlier_results}):
        return Skipped()
    return None


def _not_run_result(not_run_reason: NotRunReason) -> Outcome | Skipped:
    """What a check gives for an answer it didn't run on."""
    match not_run_reason:
        case Skipped():
            return not_run_reason
        case Unreadable():
            return unanswered_outcome(not_run_reason)
        case Blank():
            return unanswered_outcome(None)


def _check_result(results: QuestionOutcomes) -> StepResult:
    """A check's result over every question it marks."""
    outcomes = [outcome_of(result) for result in results.values()]
    output: CheckOutput = {
        "correct": bool(outcomes) and all(outcome.correct for outcome in outcomes),
        "proportion": min((outcome.proportion for outcome in outcomes), default=0.0),
        "feedback": "\n".join(outcome.feedback for outcome in outcomes if outcome.feedback),
        "marked": all(outcome.marked for outcome in outcomes),
    }
    is_skipped = bool(results) and all(isinstance(result, Skipped) for result in results.values())
    return StepResult(output, is_skipped)


def _with_feedback_and_details[R: Outcome | Skipped](
    check: Step, result: R, context: Context, question_id: QuestionId
) -> R:
    """``result`` with the check's own feedback, and the transform's answer it read."""
    if check.feedback:
        result = replace(
            result, feedback=_feedback(check, outcome_of(result), context, question_id)
        )
    if (
        isinstance(result, Skipped)
        or context.input is None
        or not isinstance(check.source, StepOutputSource)
    ):
        return result
    return replace(result, details=(Detail("Read as", context.input.written), *result.details))


def _titled(check: Step, outcome: Outcome, question_id: QuestionId) -> Outcome:
    """``outcome`` with what its breakdown says of the check: labelled by its name, and its
    question where that isn't the one marked. A named check with checks inside is titled by its
    name alone."""
    labels = tuple(
        label for label in (check.question, check.name) if label and label != question_id
    )
    title = (
        ""
        if check.name and outcome.children
        else outcome.title or check.describe().split("\n", 1)[0].removesuffix(":")
    )
    return replace(outcome, title=title, labels=(*labels, *outcome.labels))


def _feedback(check: Step, outcome: Outcome, context: Context, question_id: QuestionId) -> str:
    feedback = check.feedback
    template = feedback.template_for(outcome) if isinstance(feedback, OutcomeFeedback) else feedback
    if not template or not outcome.marked:
        return outcome.feedback
    scope = {**check.resolution.scope, "steps": dict(context.earlier_results)}
    return rendered_feedback(template, scope, outcome, context, question_id)


def rendered_feedback(
    template: str,
    scope: Mapping[str, Any],
    outcome: Outcome,
    context: Context,
    question_id: QuestionId,
) -> str:
    """``template`` filled in with ``scope`` and how ``outcome`` marked the answer."""
    return render(
        template,
        {
            **scope,
            "answer": context.written_answer(question_id).strip(),
            "correct": outcome.correct,
            "proportion": outcome.proportion,
            "feedback": outcome.feedback,
            "parts": [part.template_values(question_id) for part in outcome.parts],
        },
    )
