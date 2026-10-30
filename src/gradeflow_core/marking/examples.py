"""Marking example answers explicitly, without changing what their examples declare."""

from collections.abc import Mapping, Sequence
from math import isclose
from typing import Any

from pydantic import BaseModel

from ..common.exceptions import GradeFlowError
from ..common.model_service import ModelEndpoint
from ..methods.catalogue import BUILTINS
from ..methods.collection import Methods
from ..methods.example import Example, ExampleRow, ExpectedResult
from ..methods.method import Method
from ..methods.outcome import ROUNDING_TOLERANCE
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from .mark import mark
from .mark_scheme import MarkScheme
from .results import MarkedSheet, QuestionResult


class RowResult(MarkedSheet):
    """What marking one row gave, and whether it is as declared."""

    matches: bool | None
    """None when the row declares no result."""

    def as_example(self) -> ExampleRow:
        """The row declaring what marking gave it."""
        return ExampleRow(
            answers=self.answers,
            marks=self.marks,
            expected={
                key: ExpectedResult(marks=result.marks, feedback=result.feedback)
                for key, result in self.results.items()
            },
        )


class MarkedExample(BaseModel):
    title: str
    description: str
    rows: list[RowResult]


def evaluate_rows(
    ref: str,
    *,
    methods: Methods = BUILTINS,
    questions: QuestionSet,
    question: QuestionId | None,
    inputs: Mapping[str, Any],
    rows: Sequence[ExampleRow],
    model_endpoint: ModelEndpoint | None = None,
) -> list[RowResult]:
    """Mark each row with the method ``ref`` names, asking language models at
    ``model_endpoint``. Isolate untrusted execution.

    Answers that can't be marked stay row results. Raises ``GradeFlowError`` for settings or
    questions the method can't mark.
    """
    scheme = _scheme(ref, methods, questions, question, inputs)
    return _evaluated(scheme, questions, question, rows, model_endpoint)


def evaluate_examples(
    ref: str, *, methods: Methods = BUILTINS, model_endpoint: ModelEndpoint | None = None
) -> list[MarkedExample]:
    """Mark the examples of the method ``ref`` names, leaving what they declare unchanged."""
    method = methods.find(ref)
    return [_marked(method, example, methods, model_endpoint) for example in method.examples]


def _marked(
    method: Method[Any],
    example: Example,
    methods: Methods,
    model_endpoint: ModelEndpoint | None,
) -> MarkedExample:
    questions = QuestionSet(example.questions)
    one = len(questions) == 1 and not (method.group and not method.question_types)
    question = next(iter(questions)) if one else None
    try:
        scheme = _scheme(method.ref, methods, questions, question, example.with_)
        rows = _evaluated(scheme, questions, question, example.rows, model_endpoint)
    except GradeFlowError as error:
        raise GradeFlowError(f"{example.title}: {error}") from None
    description = scheme.markings[0].describe()
    return MarkedExample(title=example.title, description=description, rows=rows)


def _evaluated(
    scheme: MarkScheme,
    questions: QuestionSet,
    question: QuestionId | None,
    rows: Sequence[ExampleRow],
    model_endpoint: ModelEndpoint | None,
) -> list[RowResult]:
    needed = {question} if question is not None else set(questions)
    for number, row in enumerate(rows, 1):
        if missing := needed - row.answers.keys():
            raise GradeFlowError(f"Row {number} has no answer to {', '.join(sorted(missing))}.")
    marked = mark(questions, scheme, rows, model_endpoint=model_endpoint)
    return [
        RowResult(**dict(sheet), matches=_matches(row.expected, sheet.results))
        for row, sheet in zip(rows, marked, strict=True)
    ]


def _scheme(
    ref: str,
    methods: Methods,
    questions: QuestionSet,
    question: QuestionId | None,
    inputs: Mapping[str, Any],
) -> MarkScheme:
    data: dict[str, Any] = {"uses": ref, "with": dict(inputs)}
    if question is not None:
        data["question"] = question
    scheme = MarkScheme.load([data], methods=methods)
    if found := scheme.problems(questions):
        raise GradeFlowError(str(found[0]))
    return scheme


def _matches(
    expected: Mapping[QuestionId, ExpectedResult], results: Mapping[QuestionId, QuestionResult]
) -> bool | None:
    if not expected:
        return None
    return expected.keys() == results.keys() and all(
        results[key].marked
        and isclose(results[key].marks, declared.marks, abs_tol=ROUNDING_TOLERANCE)
        and results[key].feedback == declared.feedback
        for key, declared in expected.items()
    )
