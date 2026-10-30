import math
import random
from collections.abc import Iterator, Mapping, Sequence

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import markdown_join, plain_join, quoted_join, sentence
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..matching import read_or_none
from ..method import AnswerMethod
from ..outcome import CORRECT, Outcome
from ..position import Position
from ..settings import Settings, Text
from . import expressions
from .expressions import Expression

POINTS = 8
ENOUGH_POINTS = POINTS // 2
"""An accepted expression must be workable at this many points to compare answers with."""


class ExpressionSettings(Settings):
    answers: list[Text] = Field(
        min_length=1,
        title="Accepted expressions",
        examples=["2x + 2"],
        json_schema_extra=hint(accepts_answers=True),
    )


class EquivalentExpression(AnswerMethod[ExpressionSettings]):
    name = "gradeflow/equivalent-expression"
    version = 1
    label = "Equivalent Expression"
    summary = "Full marks for any expression equal to yours."
    guide = (
        "**Use it when:** The answer is an algebraic expression that can be written in many "
        "equal forms.\n\n"
        "**How it marks:** The answer and each accepted expression are worked out at several "
        "values of their variables, and an answer that agrees at every one gets full marks. "
        "Letters side by side multiply, so `2xy` is 2 × x × y. Expressions may use `+ - * / ^`, "
        "brackets, `pi`, `e`, `sqrt`, `exp`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `abs`, "
        "`ln` and `log` (base 10).\n\n"
        "**Good to know:** Every equal form is correct, so `x + x + 2` is correct for `2x + 2`.\n\n"
        "**Related:** To ask for one form, such as factorised, add a Pattern check in a "
        "Combination."
    )
    question_types = frozenset({"text"})
    settings = ExpressionSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        answers = markdown_join(self._settings_of(check).answers, conjunction="or")
        return f"Equals {answers}."

    def _mark_answer(
        self, settings: ExpressionSettings, answer: Answer, question: Question
    ) -> Outcome:
        # Accepted expressions are read as answers are, so capitals match when they're ignored.
        accepted = [
            read
            for text in settings.answers
            if (read := read_or_none(expressions.read, str(question.read(text) or text)))
            is not None
        ]
        names = frozenset().union(*(expression.names for expression in accepted))
        expected = sentence(
            f"Expected an expression equal to {quoted_join(settings.answers, conjunction='or')}"
        )
        given = read_or_none(expressions.read, str(answer))
        if given is None:
            return Outcome(0.0, f"Couldn't read it as an expression. {expected}")
        if extra := given.names - names:
            return Outcome(0.0, f"Uses {plain_join(sorted(extra), conjunction='and')}. {expected}")
        points = _points(names)
        if any(_equal(given, expression, points) for expression in accepted):
            return Outcome(1.0, CORRECT)
        return Outcome(0.0, expected)

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        texts = self._settings_of(check).answers
        accepted: dict[int, Expression] = {}
        for index, text in enumerate(texts):
            try:
                accepted[index] = expressions.read(text)
            except ValueError as error:
                yield Problem(str(error), field=("answers", index))
        points = _points(frozenset().union(*(read.names for read in accepted.values())))
        for index, expression in accepted.items():
            if len(_values(expression, points)) < ENOUGH_POINTS:
                yield Problem(
                    "Can't be worked out for most values of its variables.",
                    field=("answers", index),
                )


def _points(names: frozenset[str]) -> list[dict[str, float]]:
    """Values to compare expressions at: the same ones each time, and away from integers and zero,
    where unequal expressions tend to agree."""
    sample = random.Random(0)
    return [{name: sample.uniform(0.5, 2.5) for name in sorted(names)} for _ in range(POINTS)]


def _values(expression: Expression, points: Sequence[Mapping[str, float]]) -> dict[int, float]:
    """``expression``'s value at each of ``points`` it can be worked out at, by index."""
    values: dict[int, float] = {}
    for index, point in enumerate(points):
        try:
            values[index] = expression.value(point)
        except ValueError:
            continue
    return values


def _equal(given: Expression, accepted: Expression, points: Sequence[Mapping[str, float]]) -> bool:
    expected = _values(accepted, points)
    if len(expected) < ENOUGH_POINTS:
        return False
    actual = _values(given, points)
    return all(
        index in actual and math.isclose(actual[index], value, rel_tol=1e-6, abs_tol=1e-9)
        for index, value in expected.items()
    )
