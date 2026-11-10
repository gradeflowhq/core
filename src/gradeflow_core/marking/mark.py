import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
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

_logger = logging.getLogger(__name__)

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
    caches = [
        _MarkingCache(marking, (), {(): problem_results})  # the same for every sheet
        if (problem_results := _problem_results(marking, questions)) is not None
        else _MarkingCache(marking, tuple(sorted(marking.questions_read())), {})
        for marking in scheme.markings
    ]
    # Without language models, sheets are read one at a time, as each is marked.
    for chunk in batched(sheets, chunk_size if scheme.needs_model_endpoint() else 1, strict=False):
        answers_by_sheet = [questions.read(sheet) for sheet in chunk]
        for cache in caches:
            contexts_by_key: dict[tuple[str | None, ...], Context] = {}
            for sheet, answers in zip(chunk, answers_by_sheet, strict=True):
                cache_key = cache.key(sheet)
                if cache_key not in cache.results and cache_key not in contexts_by_key:
                    contexts_by_key[cache_key] = Context(
                        {
                            question_id: answers[question_id]
                            for question_id in cache.questions_read
                            if question_id in answers
                        },
                        questions,
                        written_answers={
                            question_id: sheet.answers[question_id]
                            for question_id in cache.questions_read
                            if question_id in sheet.answers
                        },
                        model_endpoint=model_endpoint,
                        report_progress=on_progress or (lambda _: None),
                    )
            if contexts_by_key:
                results_by_context = _marking_results(cache.marking, list(contexts_by_key.values()))
                cache.results.update(zip(contexts_by_key, results_by_context, strict=True))
        for sheet in chunk:
            results: _Results = {}
            for cache in caches:
                results = {
                    question_id: result.model_copy()
                    for question_id, result in cache.results[cache.key(sheet)].items()
                } | results
            results = _given_results(sheet, questions) | results
            result_class = MarkedSubmission if isinstance(sheet, Submission) else MarkedSheet
            yield result_class(
                **sheet.model_dump()
                | {
                    "results": {
                        question_id: results[question_id]
                        for question_id in questions
                        if question_id in results
                    },
                    "max_marks": max_marks,
                }
            )


@dataclass
class _MarkingCache:
    """A marking, the questions it reads and its results by the answers to them."""

    marking: Check
    questions_read: tuple[QuestionId, ...]
    results: dict[tuple[str | None, ...], _Results]

    def key(self, sheet: AnswerSheet) -> tuple[str | None, ...]:
        """The sheet's answers to the questions it reads."""
        return tuple(sheet.answers.get(question_id) for question_id in self.questions_read)


def _problem_results(marking: Check, questions: QuestionSet) -> _Results | None:
    """The unmarked results of a marking with problems, which say what they are; None if it has
    none."""
    problems = marking.problems(questions)
    if not problems:
        return None
    try:
        question_ids = marking.questions_marked()
    except InvalidError, TemplateError:
        question_ids = set()
    # An empty workflow marks no question, so its marking's question gets the problem.
    question_ids = question_ids or ({marking.question} if marking.question else set())
    feedback = " ".join(map(str, problems))
    return {
        question_id: _unmarked_result(questions[question_id], feedback, marking.method.label)
        for question_id in question_ids & questions.keys()
    }


def _marking_results(marking: Check, contexts: list[Context]) -> list[_Results]:
    """The results of each student in ``contexts``. A failure leaves only that student's answers
    unmarked."""
    label, questions = marking.method.label, contexts[0].questions
    question_ids = marking.questions_marked() & questions.keys()
    try:
        outcomes_by_context = marking.mark(contexts)
    except Exception as error:
        if len(contexts) > 1:
            return [_marking_results(marking, [context])[0] for context in contexts]
        if isinstance(error, GradeFlowError):
            feedback = f"Couldn't mark this answer: {error}"
        else:
            _logger.exception("Marking with %s failed", marking.uses)
            feedback = "Couldn't mark this answer."
        return [
            {
                question_id: _unmarked_result(questions[question_id], feedback, label)
                for question_id in question_ids
            }
        ]
    return [
        {
            question_id: _question_result(
                outcomes.get(question_id) or Outcome(0.0), questions[question_id], label
            )
            for question_id in question_ids
        }
        for outcomes in outcomes_by_context
    ]


def _question_result(outcome: Outcome, question: Question, label: str) -> QuestionResult:
    return QuestionResult(
        proportion=clamped(outcome.proportion),
        max_marks=question.max_marks,
        feedback=outcome.feedback,
        method_label=label,
        is_marked=outcome.marked,
        model_names=outcome.all_model_names(),
        breakdown=breakdown_of(outcome, question.max_marks),
    )


def _unmarked_result(question: Question, feedback: str, label: str) -> QuestionResult:
    return _question_result(Outcome(0.0, feedback, marked=False), question, label)


def _given_results(sheet: AnswerSheet, questions: QuestionSet) -> _Results:
    """The marks the sheet was given, as results with no method, within each question's."""
    return {
        question_id: QuestionResult(
            proportion=clamped(marks / question.max_marks) if question.max_marks else 0.0,
            max_marks=question.max_marks,
        )
        for question_id, marks in sheet.marks.items()
        if (question := questions.get(question_id)) is not None
    }
