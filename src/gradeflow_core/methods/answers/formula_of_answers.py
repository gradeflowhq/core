from collections.abc import Iterator

from natsort import natsorted
from pydantic import Field

from ...common.problems import Problem
from ...common.text import format_number, markdown_code, plain_join, plural
from ...questions.types import Answer, QuestionId, Unreadable
from ..check import Check, Context
from ..example import examples_beside
from ..matching import TOLERANCE, expecting, within
from ..method import Method
from ..outcome import CORRECT, Outcome, unanswered
from ..position import Position
from ..settings import Settings, Text, literal, specialised
from . import expressions
from .expressions import Expression


class FormulaOfAnswersSettings(Settings):
    target: QuestionId = Field(title="Question to mark")
    formula: Text = Field(
        title="Expected answer",
        description="A formula of other answers.",
        examples=["Q1 * 12"],
    )
    tolerance: float = TOLERANCE


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
        "marks only when they're right.\n\n"
        "**Related:** For known mistakes, or answers that aren't numbers, use Answer Sets."
    )
    level = "advanced"
    group = True
    settings = FormulaOfAnswersSettings
    examples = examples_beside(__file__)

    def fits(self, position: Position) -> bool:
        return position.question is None and len(_number_questions(position)) > 1

    def settings_at(self, position: Position) -> type[Settings]:
        keys = _number_questions(position)
        if not keys:
            return self.settings
        return specialised(self.settings, target={"annotation": literal(keys)})

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        formula = markdown_code(settings.formula)
        how = (
            f"is within {markdown_code(settings.tolerance)} of {formula}"
            if settings.tolerance > 0
            else f"equals {formula}"
        )
        return f"**{settings.target}** {how}, from the student's own answers."

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        settings = self._settings_of(check)
        return {settings.target: _outcome(settings, context)}

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return {self._settings_of(check).target}

    def questions_read(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        settings = self._settings_of(check)
        try:
            return {settings.target, *_read(settings.formula).names}
        except ValueError:
            return {settings.target}

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        if message := _not_a_number_question(settings.target, position):
            yield Problem(message, field=("target",))
        try:
            names = _read(settings.formula).names
        except ValueError as error:
            yield Problem(str(error), field=("formula",))
            return
        if not names:
            yield Problem("Use another answer, such as Q1 * 12.", field=("formula",))
        if settings.target in names:
            yield Problem(f"{settings.target} can't follow from itself.", field=("formula",))
        for key in natsorted(names - {settings.target}):
            if message := _not_a_number_question(key, position):
                yield Problem(message, field=("formula",))


def _read(formula: str) -> Expression:
    return expressions.read(formula, whole_names=True)


def _number_questions(position: Position) -> list[QuestionId]:
    return [key for key, question in position.questions.items() if question.type == "number"]


def _not_a_number_question(key: QuestionId, position: Position) -> str | None:
    question = position.questions.get(key)
    if question is None:
        return f"There's no question {key}."
    return None if question.type == "number" else f"{key} isn't a number question."


def _outcome(settings: FormulaOfAnswersSettings, context: Context) -> Outcome:
    answer = context.answers.get(settings.target)
    if (unread := unanswered(answer)) is not None:
        return unread
    expression = _read(settings.formula)
    keys = natsorted(expression.names)
    values: dict[str, float] = {}
    for key in keys:
        given = context.answers.get(key)
        if given is None:
            return Outcome(0.0, f"No answer to {key} to work from.")
        if isinstance(given, Unreadable):
            return Outcome(0.0, f"{key}: {given.reason}", marked=False)
        values[key] = _number(given)
    source = f"the {plural('answer', len(keys))} to {plain_join(keys, conjunction='and')}"
    try:
        expected = expression.value(values)
    except ValueError as error:
        return Outcome(0.0, f"Can't work out the expected answer from {source}. {error}")
    if within(_number(answer), expected, settings.tolerance):
        return Outcome(1.0, CORRECT)
    shown = format_number(float(f"{expected:.10g}"))
    return Outcome(0.0, f"{expecting(shown, settings.tolerance)}, from {source}.")


def _number(answer: Answer | Unreadable) -> float:
    # Problems refuse questions that aren't numbers, and unreadable answers aren't marked.
    assert isinstance(answer, int | float)
    return float(answer)
