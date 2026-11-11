from collections.abc import Iterator

from natsort import natsorted
from pydantic import Field

from ...common.problems import Problem
from ...common.text import code_span, format_number, plain_join, plural
from ...questions.types import Answer, QuestionId, Unreadable
from ...steps.context import Context
from ...steps.example import examples_beside
from ...steps.method import MarkingMethod
from ...steps.outcome import CORRECT_FEEDBACK, Outcome, unanswered_outcome
from ...steps.position import Position
from ...steps.settings import NonBlankText, Settings, literal_type, specialized
from ..matching import TOLERANCE_FIELD, expected_text, is_within_tolerance
from . import expressions
from .expressions import Expression


class FormulaOfAnswersSettings(Settings):
    question_to_mark: QuestionId = Field(title="Question to mark")
    formula: NonBlankText = Field(
        title="Expected answer",
        description="A formula of other answers.",
        examples=["Q1 * 12"],
    )
    tolerance: float = TOLERANCE_FIELD


class FormulaOfAnswers(MarkingMethod[FormulaOfAnswersSettings]):
    name = "gradeflow/formula-of-answers"
    version = 1
    label = "Formula of Answers"
    summary = "Full marks if a number follows from other answers."
    guide = (
        "**Use it when:** A number answer is worked out from earlier ones, and a student who got "
        "an earlier one wrong should still get marks for working from it.\n\n"
        "**How it marks:** Write the expected answer as a formula of other answers, such as "
        "`Q1 * 12`. Each student's own answers go into it, and their answer gets full marks if "
        "it matches, within the **Tolerance**. Formulas use the operators and functions of "
        "Equivalent Expression.\n\n"
        "**Good to know:** Write a question whose name has spaces or starts with a digit in "
        "brackets, such as `[1a]`. Mark the earlier questions on their own too, so they get "
        "marks only when they're correct.\n\n"
        "**Related:** For known mistakes, or answers that aren't numbers, use Answer Sets."
    )
    level = "advanced"
    marks_question_group = True
    settings = FormulaOfAnswersSettings
    examples = examples_beside(__file__)

    def can_run_at(self, position: Position) -> bool:
        return position.question is None and len(_number_question_ids(position)) > 1

    def settings_model_at(self, position: Position) -> type[Settings]:
        question_ids = _number_question_ids(position)
        if not question_ids:
            return self.settings
        return specialized(
            self.settings, question_to_mark={"annotation": literal_type(question_ids)}
        )

    def describe(self, settings: FormulaOfAnswersSettings) -> str:
        formula = code_span(settings.formula)
        comparison = (
            f"is within {code_span(settings.tolerance)} of {formula}"
            if settings.tolerance > 0
            else f"equals {formula}"
        )
        return f"**{settings.question_to_mark}** {comparison}, from the student's own answers."

    def mark_student(
        self, settings: FormulaOfAnswersSettings, context: Context, question_id: QuestionId | None
    ) -> dict[QuestionId, Outcome]:
        return {settings.question_to_mark: _mark_with_formula(settings, context)}

    def questions_marked(
        self, settings: FormulaOfAnswersSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        return {settings.question_to_mark}

    def questions_read(
        self, settings: FormulaOfAnswersSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        try:
            return {settings.question_to_mark, *_parse_formula(settings.formula).variables}
        except ValueError:
            return {settings.question_to_mark}

    def problems(self, settings: FormulaOfAnswersSettings, position: Position) -> Iterator[Problem]:
        if message := _number_question_message(settings.question_to_mark, position):
            yield Problem(message, field=("question_to_mark",))
        try:
            question_ids = _parse_formula(settings.formula).variables
        except ValueError as error:
            yield Problem(str(error), field=("formula",))
            return
        if not question_ids:
            yield Problem("Use another answer, such as Q1 * 12.", field=("formula",))
        if settings.question_to_mark in question_ids:
            yield Problem(
                f"{settings.question_to_mark} can't follow from itself.", field=("formula",)
            )
        for question_id in natsorted(question_ids - {settings.question_to_mark}):
            if message := _number_question_message(question_id, position):
                yield Problem(message, field=("formula",))


def _parse_formula(formula: str) -> Expression:
    return expressions.parse(formula, names_as_written=True)


def _number_question_ids(position: Position) -> list[QuestionId]:
    return [
        question_id
        for question_id, question in position.questions.items()
        if question.type == "number"
    ]


def _number_question_message(question_id: QuestionId, position: Position) -> str | None:
    question = position.questions.get(question_id)
    if question is None:
        return f"There's no question {question_id}."
    return None if question.type == "number" else f"{question_id} isn't a number question."


def _mark_with_formula(settings: FormulaOfAnswersSettings, context: Context) -> Outcome:
    answer = context.answers.get(settings.question_to_mark)
    if answer is None or isinstance(answer, Unreadable):
        return unanswered_outcome(answer)
    expression = _parse_formula(settings.formula)
    question_ids = natsorted(expression.variables)
    numbers_by_question_id: dict[str, float] = {}
    for question_id in question_ids:
        other_answer = context.answers.get(question_id)
        if other_answer is None:
            return Outcome(0.0, f"No answer to {question_id} to work from.")
        if isinstance(other_answer, Unreadable):
            return Outcome(0.0, f"{question_id}: {other_answer.reason}", marked=False)
        numbers_by_question_id[question_id] = _as_number(other_answer)
    answers_used_phrase = (
        f"the {plural('answer', len(question_ids))} "
        f"to {plain_join(question_ids, conjunction='and')}"
    )
    try:
        expected_value = expression.value(numbers_by_question_id)
    except ValueError as error:
        return Outcome(
            0.0, f"Can't work out the expected answer from {answers_used_phrase}. {error}"
        )
    if is_within_tolerance(_as_number(answer), expected_value, settings.tolerance):
        return Outcome(1.0, CORRECT_FEEDBACK)
    expected_number_text = format_number(float(f"{expected_value:.10g}"))
    return Outcome(
        0.0,
        f"{expected_text(expected_number_text, settings.tolerance)}, from {answers_used_phrase}.",
    )


def _as_number(answer: Answer | Unreadable) -> float:
    # Problems refuse questions that aren't numbers, and unreadable answers aren't marked.
    assert isinstance(answer, int | float)
    return float(answer)
