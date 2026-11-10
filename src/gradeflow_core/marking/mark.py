import logging
from collections.abc import Callable, Iterable, Iterator
from itertools import batched
from typing import overload

from ..common.exceptions import GradeFlowError, InvalidError
from ..common.model_service import ModelEndpoint
from ..methods.method import Progress
from ..methods.outcome import Outcome, clamped
from ..methods.step import Check, Context
from ..methods.templates import TemplateError
from ..questions.base import Question
from ..questions.question_set import QuestionSet
from ..questions.submissions import AnswerSheet, Submission
from ..questions.types import QuestionId
from .breakdown import breakdown_of
from .mark_scheme import MarkScheme
from .results import MarkedSheet, MarkedSubmission, QuestionResult

_log = logging.getLogger(__name__)

type _Results = dict[QuestionId, QuestionResult]


@overload
def mark(
    questions: QuestionSet,
    scheme: MarkScheme,
    sheets: Iterable[Submission],
    *,
    model_endpoint: ModelEndpoint | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    chunk_size: int = 60,
) -> Iterator[MarkedSubmission]: ...


@overload
def mark(
    questions: QuestionSet,
    scheme: MarkScheme,
    sheets: Iterable[AnswerSheet],
    *,
    model_endpoint: ModelEndpoint | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    chunk_size: int = 60,
) -> Iterator[MarkedSheet]: ...


def mark(
    questions: QuestionSet,
    scheme: MarkScheme,
    sheets: Iterable[AnswerSheet],
    *,
    model_endpoint: ModelEndpoint | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    chunk_size: int = 60,
) -> Iterator[MarkedSheet]:
    """Each sheet marked, as soon as it is; a ``Submission`` gives a ``MarkedSubmission``. Results
    follow the questions' order. A question marked twice, which the mark scheme's problems
    report, gets its first marking's result; one no marking marks keeps the marks the sheet was
    given, if any; one not in ``questions`` gets none.

    A marking sees only the answers it reads, so sheets with the same answers to those are
    marked once. A marking that asks a language model at ``model_endpoint`` takes ``chunk_size``
    sheets at a time, and says how far it got to ``on_progress``."""
    max_marks = sum(question.max_marks for question in questions.values())
    # Each marking with the questions it reads and its results by the answers to them.
    cached_markings: list[
        tuple[Check, tuple[QuestionId, ...], dict[tuple[str | None, ...], _Results]]
    ] = []
    markings_without_problems = []
    for marking in scheme.markings:
        if (problem_results := _problem_results(marking, questions)) is not None:
            cached_markings.append((marking, (), {(): problem_results}))  # the same for every sheet
        else:
            cached_markings.append((marking, tuple(sorted(marking.questions_read())), {}))
            markings_without_problems.append(marking)
    needs_batches = any(marking.needs_model_endpoint() for marking in markings_without_problems)
    # Without language models, sheets are read one at a time, as each is marked.
    for chunk in batched(sheets, chunk_size if needs_batches else 1, strict=False):
        read_sheets = [questions.read(sheet) for sheet in chunk]
        for marking, marking_questions_read, cache in cached_markings:
            contexts_by_cache_key: dict[tuple[str | None, ...], Context] = {}
            for sheet, answers in zip(chunk, read_sheets, strict=True):
                cache_key = tuple(sheet.answers.get(key) for key in marking_questions_read)
                if cache_key not in cache and cache_key not in contexts_by_cache_key:
                    contexts_by_cache_key[cache_key] = Context(
                        {key: answers[key] for key in marking_questions_read if key in answers},
                        questions,
                        written_answers={
                            key: sheet.answers[key]
                            for key in marking_questions_read
                            if key in sheet.answers
                        },
                        model_endpoint=model_endpoint,
                        report_progress=on_progress or (lambda _: None),
                    )
            if contexts_by_cache_key:
                results_by_student = _marking_results(marking, list(contexts_by_cache_key.values()))
                cache.update(zip(contexts_by_cache_key, results_by_student, strict=True))
        for sheet in chunk:
            results: dict[QuestionId, QuestionResult] = {}
            for _, marking_questions_read, cache in cached_markings:
                cache_key = tuple(sheet.answers.get(key) for key in marking_questions_read)
                results = {
                    key: result.model_copy() for key, result in cache[cache_key].items()
                } | results
            results = _given_results(sheet, questions) | results
            result_class = MarkedSubmission if isinstance(sheet, Submission) else MarkedSheet
            yield result_class(
                **sheet.model_dump(include=set(result_class.model_fields))
                | {
                    "results": {key: results[key] for key in questions if key in results},
                    "max_marks": max_marks,
                }
            )


def _problem_results(marking: Check, questions: QuestionSet) -> _Results | None:
    """The unmarked results of a marking with problems, which say what they are; None if it has
    none."""
    problems = marking.problems(questions)
    if not problems:
        return None
    try:
        keys = marking.questions_marked()
    except InvalidError, TemplateError:
        keys = set()
    # An empty workflow marks no question, so its marking's question gets the problem.
    keys = keys or ({marking.question} if marking.question else set())
    feedback = " ".join(map(str, problems))
    return {
        key: _unmarked_result(questions[key], feedback, marking.method.label)
        for key in keys & questions.keys()
    }


def _marking_results(marking: Check, contexts: list[Context]) -> list[_Results]:
    """The results of each student in ``contexts``. A failure leaves only that student's answers
    unmarked."""
    label, questions = marking.method.label, contexts[0].questions
    keys = marking.questions_marked() & questions.keys()
    try:
        outcomes_by_student = marking.mark(contexts)
    except Exception as error:
        if len(contexts) > 1:
            return [_marking_results(marking, [context])[0] for context in contexts]
        if isinstance(error, GradeFlowError):
            feedback = f"Couldn't mark this answer: {error}"
        else:
            _log.exception("Marking with %s failed", marking.uses)
            feedback = "Couldn't mark this answer."
        return [{key: _unmarked_result(questions[key], feedback, label) for key in keys}]
    return [
        {key: _result(outcomes.get(key) or Outcome(0.0), questions[key], label) for key in keys}
        for outcomes in outcomes_by_student
    ]


def _result(outcome: Outcome, question: Question, label: str) -> QuestionResult:
    return QuestionResult(
        proportion=clamped(outcome.proportion),
        max_marks=question.max_marks,
        feedback=outcome.feedback,
        method_label=outcome.method or label,
        is_marked=outcome.marked,
        model_names=outcome.all_models(),
        breakdown=breakdown_of(outcome, question.max_marks),
    )


def _unmarked_result(question: Question, feedback: str, label: str) -> QuestionResult:
    return _result(Outcome(0.0, feedback, marked=False), question, label)


def _given_results(sheet: AnswerSheet, questions: QuestionSet) -> dict[QuestionId, QuestionResult]:
    """The marks the sheet was given, as results with no method, within each question's."""
    return {
        key: QuestionResult(
            proportion=clamped(marks / question.max_marks) if question.max_marks else 0.0,
            max_marks=question.max_marks,
        )
        for key, marks in sheet.marks.items()
        if (question := questions.get(key)) is not None
    }
