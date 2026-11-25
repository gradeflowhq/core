import itertools
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from typing import Annotated

from pydantic import Field

from ...common.hints import hint
from ...common.paths import pointer
from ...common.problems import Problem
from ...common.text import (
    as_sentence,
    code_join,
    markdown_code_list,
    plain_join,
    quoted,
    quoted_join,
    reads_better_listed,
)
from ...questions.base import Question
from ...questions.text import TextQuestion
from ...questions.types import Answer
from ...steps.example import examples_beside
from ...steps.method import AnswerMethod
from ...steps.outcome import CORRECT_FEEDBACK, Outcome
from ...steps.position import Position
from ...steps.settings import IDENTIFIER, NonBlankText, Settings
from ..matching import read_or_none
from . import expressions
from .expressions import Expression

SAMPLE_POINT_COUNT = 16
MIN_WORKABLE_POINTS = SAMPLE_POINT_COUNT // 4
"""An accepted expression must be workable at this many points to compare answers with."""


class EquivalentExpressionSettings(Settings):
    answers: list[NonBlankText] = Field(
        min_length=1,
        title="Accepted expressions",
        examples=["2x + 2"],
        json_schema_extra=hint(lists_answers=True),
    )
    variables: list[Annotated[str, Field(pattern=IDENTIFIER)]] = Field(
        default_factory=list,
        title="Variables",
        description="Names to read as one variable, such as u1.",
        json_schema_extra=hint(item_label="variable"),
    )


class EquivalentExpression(AnswerMethod[EquivalentExpressionSettings]):
    name = "gradeflow/equivalent-expression"
    version = 1
    label = "Equivalent Expression"
    summary = "Correct if the expression equals one of yours."
    guide = (
        "For algebra answers that can be written in many equal forms, such as `2(x + 1)` and `2x + "
        "2`.\n\n"
        "The answer and each accepted expression are worked out at several values of their "
        "variables. If they agree every time, the answer is correct. Every equal form passes, so "
        "`x + x + 2` is correct for `2x + 2`.\n\n"
        "Expressions use `+ - * / ^` and brackets. Names split into single letters and numbers, "
        "which multiply: `2xy` is 2 × x × y, and `u2v` is u × 2 × v. A name with an "
        "underscore, like `u_1`, a Greek letter's name, like `alpha`, and a name in **Variables** "
        "are one variable. You can use `pi`, `e`, `sqrt`, `exp`, `abs`, `ln`, `log` (base 10), "
        "`sin`, `cos`, `tan`, `asin`, `acos` and `atan`. An answer that can't be read needs "
        "marking.\n\n"
        "To also require one form, such as factorised, add a Pattern check beside it in a "
        "Workflow."
    )
    question_types = frozenset({"text"})
    settings = EquivalentExpressionSettings
    examples = examples_beside(__file__)

    def describe(self, settings: EquivalentExpressionSettings) -> str:
        answers = settings.answers
        variables_note = (
            f", with variables {plain_join(settings.variables, conjunction='and')}"
            if settings.variables
            else ""
        )
        if reads_better_listed(answers):
            return f"Equals any of these{variables_note}:\n{markdown_code_list(answers)}"
        return f"Equals {code_join(answers, conjunction='or')}{variables_note}."

    def mark_answer(
        self, settings: EquivalentExpressionSettings, answer: Answer, question: Question
    ) -> Outcome:
        variables = _variables(settings, question)
        accepted_expressions = [
            _accepted_expression(expression_text, question, variables)
            for expression_text in settings.answers
        ]
        accepted_variables = frozenset().union(
            *(expression.variables for expression in accepted_expressions)
        )
        expected_sentence = as_sentence(
            f"Expected an expression equal to {quoted_join(settings.answers, conjunction='or')}"
        )
        answer_expression = read_or_none(
            partial(expressions.parse, variables=variables), str(answer)
        )
        if answer_expression is None:
            return Outcome(0.0, "Couldn't read it as an expression.", marked=False)
        if unknown_variables := answer_expression.variables - accepted_variables:
            variable_names = plain_join(sorted(unknown_variables), conjunction="and")
            return Outcome(0.0, f"Uses {variable_names}. {expected_sentence}")
        points = _sample_points(accepted_variables)
        if any(
            _is_equivalent(answer_expression, expression, points)
            for expression in accepted_expressions
        ):
            return Outcome(1.0, CORRECT_FEEDBACK)
        return Outcome(0.0, expected_sentence)

    def problems(
        self, settings: EquivalentExpressionSettings, position: Position
    ) -> Iterator[Problem]:
        question = position.question_model
        assert isinstance(question, TextQuestion)  # it marks only text questions
        if question.reading.ignore_punctuation:
            yield Problem(
                "The question ignores punctuation, which removes the brackets and signs "
                "expressions need."
            )
            return
        for index, variable in enumerate(settings.variables):
            if variable in expressions.RESERVED_NAMES:
                yield Problem(
                    f"{quoted(variable)} is a function or constant.",
                    field=pointer("variables", index),
                )
        variables = _variables(settings, question)
        accepted_expressions_by_index: dict[int, Expression] = {}
        for index, expression_text in enumerate(settings.answers):
            try:
                accepted_expressions_by_index[index] = _accepted_expression(
                    expression_text, question, variables
                )
            except ValueError as error:
                yield Problem(str(error), field=pointer("answers", index))
        points = _sample_points(
            frozenset().union(
                *(expression.variables for expression in accepted_expressions_by_index.values())
            )
        )
        for index, expression in accepted_expressions_by_index.items():
            if len(_values_at_points(expression, points)) < MIN_WORKABLE_POINTS:
                yield Problem(
                    "Can't be worked out for enough values of its variables.",
                    field=pointer("answers", index),
                )


def _variables(settings: EquivalentExpressionSettings, question: Question) -> frozenset[str]:
    """The listed variables, read as answers to ``question`` are."""
    return frozenset(str(question.read(variable) or variable) for variable in settings.variables)


def _accepted_expression(
    expression_text: str, question: Question, variables: frozenset[str]
) -> Expression:
    """An accepted expression, read as answers to ``question`` are, so capitals match when they're
    ignored."""
    return expressions.parse(
        str(question.read(expression_text) or expression_text), variables=variables
    )


def _sample_points(variables: frozenset[str]) -> list[dict[str, float]]:
    """Points to compare expressions at, the same each time and away from integers and zero, where
    unequal expressions tend to agree. Half are positive, and the rest have each sign combination of
    up to three variables."""
    generator = random.Random(0)
    sign_patterns = [(1, 1, 1)] * (SAMPLE_POINT_COUNT // 2) + [
        signs[::-1] for signs in itertools.product((1, -1), repeat=3)
    ]
    return [
        {
            variable: generator.uniform(0.5, 2.5) * signs[variable_index % 3]
            for variable_index, variable in enumerate(sorted(variables))
        }
        for signs in sign_patterns
    ]


def _values_at_points(
    expression: Expression, points: Sequence[Mapping[str, float]]
) -> dict[int, float]:
    """``expression``'s value at each of ``points`` it can be worked out at, by index."""
    values_by_point_index: dict[int, float] = {}
    for index, point in enumerate(points):
        try:
            values_by_point_index[index] = expression.value(point)
        except ValueError:
            continue
    return values_by_point_index


def _is_equivalent(
    answer_expression: Expression,
    accepted_expression: Expression,
    points: Sequence[Mapping[str, float]],
) -> bool:
    accepted_values = _values_at_points(accepted_expression, points)
    answer_values = _values_at_points(answer_expression, points)
    return all(
        index in answer_values
        and math.isclose(answer_values[index], value, rel_tol=1e-6, abs_tol=1e-9)
        for index, value in accepted_values.items()
    )
