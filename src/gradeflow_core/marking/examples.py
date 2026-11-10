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
from ..methods.outcome import NO_ANSWER, ROUNDING_TOLERANCE, Skipped, StepResult
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
                key: ExpectedResult(marks=result.marks, feedback=result.feedback)
                for key, result in self.results.items()
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
    method = methods.find(reference)
    if not method.gives_marks(settings):
        return _transform_row_results(method, methods, questions, question, settings, rows)
    scheme = _single_marking_scheme(reference, methods, questions, question, settings)
    return _row_results(scheme, questions, question, rows, model_endpoint)


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
    question = next(iter(questions)) if marks_one_question else None
    try:
        rows = mark_rows(
            method.ref,
            methods=methods,
            questions=questions,
            question=question,
            settings=example.with_,
            rows=example.rows,
            model_endpoint=model_endpoint,
        )
    except GradeFlowError as error:
        raise GradeFlowError(f"{example.title}: {error}") from None
    description = _load_step(method, methods, question, example.with_).describe()
    return MarkedExample(title=example.title, description=description, rows=rows)


def _step_data(
    reference: str, question: QuestionId | None, settings: Mapping[str, Any]
) -> dict[str, Any]:
    data = {"uses": reference, "with": dict(settings)}
    return {"question": question, **data} if question is not None else data


def _load_step(
    method: Method[Any],
    methods: Methods,
    question: QuestionId | None,
    settings: Mapping[str, Any],
) -> Step:
    return Step.load(_step_data(method.ref, question, settings), methods=methods)


def _transform_row_results(
    method: Method[Any],
    methods: Methods,
    questions: QuestionSet,
    question: QuestionId | None,
    settings: Mapping[str, Any],
    rows: Sequence[ExampleRow],
) -> list[RowResult]:
    """What the transform gives each row, as a result that says what it read."""
    step = _load_step(method, methods, question, settings)
    assert isinstance(step, Transform)  # ``mark_rows`` sends only methods that give no marks
    if problems := step.problems(questions):
        raise GradeFlowError(str(problems[0]))
    contexts = [
        Context(questions.read(row), questions, written_answers=row.answers) for row in rows
    ]
    max_marks = sum(question_model.max_marks for question_model in questions.values())
    key = step.question_key(question)
    keys = {key} if key else step.questions_read(question)
    row_results = []
    for row, result in zip(rows, step.transform(contexts, question), strict=True):
        results = _output_results(result, keys, questions, method.label)
        row_results.append(
            RowResult(
                answers=row.answers,
                marks=row.marks,
                results=results,
                max_marks=max_marks,
                matches_expected=_matches(row.expected, results),
            )
        )
    return row_results


def _output_results(
    result: StepResult, keys: Collection[QuestionId], questions: QuestionSet, label: str
) -> dict[QuestionId, QuestionResult]:
    answer = result.transform_result
    if isinstance(answer, Skipped):
        return {}
    proportion, feedback = 0.0, NO_ANSWER
    if isinstance(answer, Unreadable):
        feedback = answer.reason
    elif answer is not None:
        proportion, feedback = 1.0, f"Read as: {write_answer(answer)}"
    return {
        key: QuestionResult(
            proportion=proportion,
            max_marks=questions[key].max_marks,
            feedback=feedback,
            method_label=label,
            is_marked=not isinstance(answer, Unreadable),
        )
        for key in keys
    }


def _row_results(
    scheme: MarkScheme,
    questions: QuestionSet,
    question: QuestionId | None,
    rows: Sequence[ExampleRow],
    model_endpoint: ModelEndpoint | None,
) -> list[RowResult]:
    needed_questions = {question} if question is not None else set(questions)
    for number, row in enumerate(rows, 1):
        if missing := needed_questions - row.answers.keys():
            raise GradeFlowError(f"Row {number} has no answer to {', '.join(sorted(missing))}.")
    marked_submissions = mark(questions, scheme, rows, model_endpoint=model_endpoint)
    return [
        RowResult(**dict(sheet), matches_expected=_matches(row.expected, sheet.results))
        for row, sheet in zip(rows, marked_submissions, strict=True)
    ]


def _single_marking_scheme(
    reference: str,
    methods: Methods,
    questions: QuestionSet,
    question: QuestionId | None,
    settings: Mapping[str, Any],
) -> MarkScheme:
    scheme = MarkScheme.load([_step_data(reference, question, settings)], methods=methods)
    if problems := scheme.problems(questions):
        raise GradeFlowError(str(problems[0]))
    return scheme


def _matches(
    expected: Mapping[QuestionId, ExpectedResult], results: Mapping[QuestionId, QuestionResult]
) -> bool | None:
    if not expected:
        return None
    return expected.keys() == results.keys() and all(
        results[key].is_marked
        and isclose(results[key].marks, expected_result.marks, abs_tol=ROUNDING_TOLERANCE)
        and results[key].feedback == expected_result.feedback
        for key, expected_result in expected.items()
    )
