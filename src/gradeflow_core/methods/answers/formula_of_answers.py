from collections.abc import Iterator

from natsort import natsorted
from pydantic import Field

from ...common.problems import Problem
from ...common.text import code_span, format_number, plain_join, plural
from ...questions.types import Answer, QuestionId, Unreadable
from ..check import Check, Context
from ..example import examples_beside
from ..matching import TOLERANCE_FIELD, expected_text, within
from ..method import Method
from ..outcome import CORRECT, Outcome, unanswered_outcome
from ..position import Position
from ..settings import NonBlankText, Settings, literal, specialized
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


class FormulaOfAnswers(Method[FormulaOfAnswersSettings]):
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

    def can_mark_at(self, position: Position) -> bool:
        return position.question is None and len(_number_questions(position)) > 1

    def settings_model_at(self, position: Position) -> type[Settings]:
        keys = _number_questions(position)
        if not keys:
            return self.settings
        return specialized(self.settings, question_to_mark={"annotation": literal(keys)})

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        formula = code_span(settings.formula)
        comparison = (
            f"is within {code_span(settings.tolerance)} of {formula}"
            if settings.tolerance > 0
            else f"equals {formula}"
        )
        return f"**{settings.question_to_mark}** {comparison}, from the student's own answers."

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        settings = self._settings_of(check)
        return {settings.question_to_mark: _mark_with_formula(settings, context)}

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return {self._settings_of(check).question_to_mark}

    def questions_read(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        settings = self._settings_of(check)
        try:
            return {settings.question_to_mark, *_parse_formula(settings.formula).variables}
        except ValueError:
            return {settings.question_to_mark}

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        if message := _number_question_problem(settings.question_to_mark, position):
            yield Problem(message, field=("question_to_mark",))
        try:
            question_keys = _parse_formula(settings.formula).variables
        except ValueError as error:
            yield Problem(str(error), field=("formula",))
            return
        if not question_keys:
            yield Problem("Use another answer, such as Q1 * 12.", field=("formula",))
        if settings.question_to_mark in question_keys:
            yield Problem(
                f"{settings.question_to_mark} can't follow from itself.", field=("formula",)
            )
        for key in natsorted(question_keys - {settings.question_to_mark}):
            if message := _number_question_problem(key, position):
                yield Problem(message, field=("formula",))


def _parse_formula(formula: str) -> Expression:
    return expressions.parse(formula, names_as_written=True)


def _number_questions(position: Position) -> list[QuestionId]:
    return [key for key, question in position.questions.items() if question.type == "number"]


def _number_question_problem(key: QuestionId, position: Position) -> str | None:
    question = position.questions.get(key)
    if question is None:
        return f"There's no question {key}."
    return None if question.type == "number" else f"{key} isn't a number question."


def _mark_with_formula(settings: FormulaOfAnswersSettings, context: Context) -> Outcome:
    answer = context.answers.get(settings.question_to_mark)
    if (outcome := unanswered_outcome(answer)) is not None:
        return outcome
    expression = _parse_formula(settings.formula)
    keys = natsorted(expression.variables)
    values: dict[str, float] = {}
    for key in keys:
        other_answer = context.answers.get(key)
        if other_answer is None:
            return Outcome(0.0, f"No answer to {key} to work from.")
        if isinstance(other_answer, Unreadable):
            return Outcome(0.0, f"{key}: {other_answer.reason}", marked=False)
        values[key] = _as_number(other_answer)
    answers_used = f"the {plural('answer', len(keys))} to {plain_join(keys, conjunction='and')}"
    try:
        expected = expression.value(values)
    except ValueError as error:
        return Outcome(0.0, f"Can't work out the expected answer from {answers_used}. {error}")
    if within(_as_number(answer), expected, settings.tolerance):
        return Outcome(1.0, CORRECT)
    expected_number = format_number(float(f"{expected:.10g}"))
    return Outcome(
        0.0, f"{expected_text(expected_number, settings.tolerance)}, from {answers_used}."
    )


def _as_number(answer: Answer | Unreadable) -> float:
    # Problems refuse questions that aren't numbers, and unreadable answers aren't marked.
    assert isinstance(answer, int | float)
    return float(answer)
