import math
import random
from collections.abc import Iterator, Mapping, Sequence

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import (
    as_sentence,
    code_join,
    markdown_code_list,
    plain_join,
    quoted_join,
    reads_better_listed,
)
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..matching import read_or_none
from ..method import AnswerMethod
from ..outcome import CORRECT, Outcome
from ..position import Position
from ..settings import NonBlankText, Settings
from . import expressions
from .expressions import Expression

SAMPLE_POINT_COUNT = 8
MIN_WORKABLE_POINTS = SAMPLE_POINT_COUNT // 2
"""An accepted expression must be workable at this many points to compare answers with."""


class EquivalentExpressionSettings(Settings):
    answers: list[NonBlankText] = Field(
        min_length=1,
        title="Accepted expressions",
        examples=["2x + 2"],
        json_schema_extra=hint(lists_answers=True),
    )


class EquivalentExpression(AnswerMethod[EquivalentExpressionSettings]):
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
    settings = EquivalentExpressionSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        answers = self._settings_of(check).answers
        if reads_better_listed(answers):
            return f"Equals any of these:\n{markdown_code_list(answers)}"
        return f"Equals {code_join(answers, conjunction='or')}."

    def _mark_answer(
        self, settings: EquivalentExpressionSettings, answer: Answer, question: Question
    ) -> Outcome:
        # Accepted expressions are read as answers are, so capitals match when they're ignored.
        accepted = [
            expression
            for text in settings.answers
            if (expression := read_or_none(expressions.parse, str(question.read(text) or text)))
            is not None
        ]
        variables = frozenset().union(*(expression.variables for expression in accepted))
        expected = as_sentence(
            f"Expected an expression equal to {quoted_join(settings.answers, conjunction='or')}"
        )
        answer_expression = read_or_none(expressions.parse, str(answer))
        if answer_expression is None:
            return Outcome(0.0, f"Couldn't read it as an expression. {expected}")
        if unknown_variables := answer_expression.variables - variables:
            variable_names = plain_join(sorted(unknown_variables), conjunction="and")
            return Outcome(0.0, f"Uses {variable_names}. {expected}")
        points = _sample_points(variables)
        if any(_is_equivalent(answer_expression, expression, points) for expression in accepted):
            return Outcome(1.0, CORRECT)
        return Outcome(0.0, expected)

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        texts = self._settings_of(check).answers
        accepted: dict[int, Expression] = {}
        for index, text in enumerate(texts):
            try:
                accepted[index] = expressions.parse(text)
            except ValueError as error:
                yield Problem(str(error), field=("answers", index))
        points = _sample_points(
            frozenset().union(*(expression.variables for expression in accepted.values()))
        )
        for index, expression in accepted.items():
            if len(_values_at_points(expression, points)) < MIN_WORKABLE_POINTS:
                yield Problem(
                    "Can't be worked out for most values of its variables.",
                    field=("answers", index),
                )


def _sample_points(variables: frozenset[str]) -> list[dict[str, float]]:
    """Values to compare expressions at: the same ones each time, and away from integers and zero,
    where unequal expressions tend to agree."""
    generator = random.Random(0)
    return [
        {variable: generator.uniform(0.5, 2.5) for variable in sorted(variables)}
        for _ in range(SAMPLE_POINT_COUNT)
    ]


def _values_at_points(
    expression: Expression, points: Sequence[Mapping[str, float]]
) -> dict[int, float]:
    """``expression``'s value at each of ``points`` it can be worked out at, by index."""
    values: dict[int, float] = {}
    for index, point in enumerate(points):
        try:
            values[index] = expression.value(point)
        except ValueError:
            continue
    return values


def _is_equivalent(
    answer_expression: Expression, accepted: Expression, points: Sequence[Mapping[str, float]]
) -> bool:
    expected = _values_at_points(accepted, points)
    if len(expected) < MIN_WORKABLE_POINTS:
        return False
    actual = _values_at_points(answer_expression, points)
    return all(
        index in actual and math.isclose(actual[index], value, rel_tol=1e-6, abs_tol=1e-9)
        for index, value in expected.items()
    )
