"""Marking example answers explicitly, without changing what their examples declare."""

from collections.abc import Collection, Mapping, Sequence
from math import isclose
from typing import Any

from pydantic import BaseModel

from ..common.exceptions import GradeFlowError
from ..common.model_service import ModelEndpoint
from ..methods.catalogue import BUILTINS
from ..methods.collection import Methods
from ..methods.example import Example, ExampleRow, ExpectedResult
from ..methods.method import Method
from ..methods.outcome import NO_ANSWER_FEEDBACK, ROUNDING_TOLERANCE, SkippedStep, StepResult
from ..methods.step import Context, Step, Transform
from ..questions.question_set import QuestionSet
from ..questions.reading import write_answer
from ..questions.types import QuestionId, Unreadable
from .mark import mark
from .mark_scheme import MarkScheme
from .results import MarkedSheet, QuestionResult


class RowResult(MarkedSheet):
    """What marking one row gave, and whether it is as declared."""

    matches_expected: bool | None
    """None when the row declares no result."""

    def as_example(self) -> ExampleRow:
        """The row declaring what marking gave it."""
        return ExampleRow(
            answers=self.answers,
            marks=self.marks,
            expected={
                question_id: ExpectedResult(marks=result.marks, feedback=result.feedback)
                for question_id, result in self.results.items()
            },
        )


class MarkedExample(BaseModel):
    title: str
    description: str
    rows: list[RowResult]


def mark_rows(
    reference: str,
    *,
    methods: Methods = BUILTINS,
    questions: QuestionSet,
    question: QuestionId | None,
    settings: Mapping[str, Any],
    rows: Sequence[ExampleRow],
    model_endpoint: ModelEndpoint | None = None,
) -> list[RowResult]:
    """Mark each row with the method ``reference`` names, asking language models at
    ``model_endpoint``. Isolate untrusted execution.

    Answers that can't be marked stay row results. Raises ``GradeFlowError`` for settings or
    questions the method can't mark.
    """
    data = method_step_data(reference, question, settings)
    if methods.find(reference).gives_marks(settings):
        scheme = MarkScheme.load([data], methods=methods)
        if problems := scheme.problems(questions):
            raise GradeFlowError(str(problems[0]))
        return _row_results(scheme, questions, question, rows, model_endpoint)
    step = Step.load(data, methods=methods)
    assert isinstance(step, Transform)  # a method that gives no marks
    if problems := step.problems(questions):
        raise GradeFlowError(str(problems[0]))
    return _transform_row_results(step, questions, question, rows)


def mark_examples(
    reference: str, *, methods: Methods = BUILTINS, model_endpoint: ModelEndpoint | None = None
) -> list[MarkedExample]:
    """Mark the examples of the method ``reference`` names, leaving what they declare unchanged."""
    method = methods.find(reference)
    return [
        _marked_example(method, example, methods, model_endpoint) for example in method.examples
    ]


def _marked_example(
    method: Method[Any],
    example: Example,
    methods: Methods,
    model_endpoint: ModelEndpoint | None,
) -> MarkedExample:
    questions = QuestionSet(example.questions)
    marks_only_groups = method.marks_question_group and not method.question_types
    marks_one_question = len(questions) == 1 and not marks_only_groups
    question_id = next(iter(questions)) if marks_one_question else None
    try:
        row_results = mark_rows(
            method.ref,
            methods=methods,
            questions=questions,
            question=question_id,
            settings=example.with_,
            rows=example.rows,
            model_endpoint=model_endpoint,
        )
    except GradeFlowError as error:
        raise GradeFlowError(f"{example.title}: {error}") from None
    step = Step.load(method_step_data(method.ref, question_id, example.with_), methods=methods)
    return MarkedExample(title=example.title, description=step.describe(), rows=row_results)


def method_step_data(
    reference: str, question_id: QuestionId | None, settings: Mapping[str, Any]
) -> dict[str, Any]:
    data = {"uses": reference, "with": dict(settings)}
    return {"question": question_id, **data} if question_id is not None else data


def _transform_row_results(
    step: Transform,
    questions: QuestionSet,
    question_id: QuestionId | None,
    rows: Sequence[ExampleRow],
) -> list[RowResult]:
    """What the transform gives each row, as a result that says what it read."""
    contexts = [
        Context(questions.read(row), questions, written_answers=row.answers) for row in rows
    ]
    max_marks = sum(question.max_marks for question in questions.values())
    marked_question_id = step.marked_question_id(question_id)
    question_ids = {marked_question_id} if marked_question_id else step.questions_read(question_id)
    row_results = []
    for row, result in zip(rows, step.transform(contexts, question_id), strict=True):
        results = _transform_results(result, question_ids, questions, step.method.label)
        row_results.append(
            RowResult(
                answers=row.answers,
                marks=row.marks,
                results=results,
                max_marks=max_marks,
                matches_expected=_matches_expected(row.expected, results),
            )
        )
    return row_results


def _transform_results(
    result: StepResult, question_ids: Collection[QuestionId], questions: QuestionSet, label: str
) -> dict[QuestionId, QuestionResult]:
    answer = result.transform_result
    if isinstance(answer, SkippedStep):
        return {}
    proportion, feedback = 0.0, NO_ANSWER_FEEDBACK
    if isinstance(answer, Unreadable):
        feedback = answer.reason
    elif answer is not None:
        proportion, feedback = 1.0, f"Read as: {write_answer(answer)}"
    return {
        question_id: QuestionResult(
            proportion=proportion,
            max_marks=questions[question_id].max_marks,
            feedback=feedback,
            method_label=label,
            is_marked=not isinstance(answer, Unreadable),
        )
        for question_id in question_ids
    }


def _row_results(
    scheme: MarkScheme,
    questions: QuestionSet,
    question_id: QuestionId | None,
    rows: Sequence[ExampleRow],
    model_endpoint: ModelEndpoint | None,
) -> list[RowResult]:
    needed_question_ids = {question_id} if question_id is not None else set(questions)
    for row_number, row in enumerate(rows, 1):
        if missing_question_ids := needed_question_ids - row.answers.keys():
            raise GradeFlowError(
                f"Row {row_number} has no answer to {', '.join(sorted(missing_question_ids))}."
            )
    marked_sheets = mark(questions, scheme, rows, model_endpoint=model_endpoint)
    return [
        RowResult(**dict(sheet), matches_expected=_matches_expected(row.expected, sheet.results))
        for row, sheet in zip(rows, marked_sheets, strict=True)
    ]


def _matches_expected(
    expected_results: Mapping[QuestionId, ExpectedResult],
    results: Mapping[QuestionId, QuestionResult],
) -> bool | None:
    if not expected_results:
        return None
    return expected_results.keys() == results.keys() and all(
        results[question_id].is_marked
        and isclose(results[question_id].marks, expected_result.marks, abs_tol=ROUNDING_TOLERANCE)
        and results[question_id].feedback == expected_result.feedback
        for question_id, expected_result in expected_results.items()
    )
