import logging
from collections.abc import Callable, Iterable, Iterator
from itertools import batched
from typing import Any, overload

from ..common.exceptions import GradeFlowError, InvalidError
from ..common.model_service import ModelEndpoint
from ..methods.check import Check, Context, checks_in
from ..methods.method import Method, Progress
from ..methods.outcome import Outcome, clamped
from ..methods.templates import TemplateError
from ..questions.base import Question
from ..questions.question_set import QuestionSet, read_answer
from ..questions.submissions import AnswerSheet, Submission
from ..questions.types import QuestionId, Unreadable
from .breakdown import breakdown_of
from .mark_scheme import MarkScheme
from .results import MarkedSheet, MarkedSubmission, QuestionResult

_log = logging.getLogger(__name__)


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
    marked once. Methods that work out something for many answers at once, such as asking a
    language model at ``model_endpoint``, do so for ``chunk_size`` sheets at a time, and say how
    far they got to ``on_progress``."""
    problem_results = [_problem_results(marking, questions) for marking in scheme.markings]
    questions_read = [_questions_read(marking) for marking in scheme.markings]
    caches: list[dict[tuple[str | None, ...], dict[QuestionId, QuestionResult]]] = [
        {} for _ in scheme.markings
    ]
    precomputing_checks = _precomputing_checks(scheme, problem_results)
    precomputed: dict[str, dict[QuestionId, dict[Any, Any]]] = {}
    precomputed_answers: set[tuple[str, QuestionId, str]] = set()
    # Without methods that precompute, sheets are read one at a time, as each is marked.
    for chunk in batched(sheets, chunk_size if precomputing_checks else 1, strict=False):
        read_sheets = [questions.read(sheet) for sheet in chunk]
        for (method, question), checks in precomputing_checks.items():
            texts = {
                sheet.answers[question].strip(): None
                for sheet, answers in zip(chunk, read_sheets, strict=True)
                if (answer := read_answer(answers, question, None)) is not None
                and not isinstance(answer, Unreadable)
                and (method.ref, question, sheet.answers[question].strip())
                not in precomputed_answers
            }
            if not texts:
                continue
            precomputed_answers.update((method.ref, question, text) for text in texts)
            precomputed_values = method.precompute(
                checks,
                list(texts),
                questions[question],
                model_endpoint,
                on_progress or (lambda _: None),
            )
            precomputed.setdefault(method.ref, {}).setdefault(question, {}).update(
                precomputed_values
            )
        for sheet, answers in zip(chunk, read_sheets, strict=True):
            results: dict[QuestionId, QuestionResult] = {}
            for marking, marking_problem_results, marking_questions_read, cache in zip(
                scheme.markings, problem_results, questions_read, caches, strict=True
            ):
                if marking_problem_results is not None:
                    marking_results = marking_problem_results
                elif marking_questions_read is None:
                    context = Context(
                        answers, questions, written_answers=sheet.answers, precomputed=precomputed
                    )
                    marking_results = _marking_results(marking, context)
                else:
                    cache_key = tuple(sheet.answers.get(key) for key in marking_questions_read)
                    if cache_key not in cache:
                        read_answers = {
                            key: answers[key] for key in marking_questions_read if key in answers
                        }
                        written_answers = {
                            key: sheet.answers[key]
                            for key in marking_questions_read
                            if key in sheet.answers
                        }
                        context = Context(
                            read_answers,
                            questions,
                            written_answers=written_answers,
                            precomputed=precomputed,
                        )
                        cache[cache_key] = _marking_results(marking, context)
                    marking_results = cache[cache_key]
                results = {
                    key: result.model_copy() for key, result in marking_results.items()
                } | results
            results = _given_results(sheet, questions) | results
            result_class = MarkedSubmission if isinstance(sheet, Submission) else MarkedSheet
            yield result_class(
                **sheet.model_dump(include=set(result_class.model_fields)),
                results={key: results[key] for key in questions if key in results},
            )


def _precomputing_checks(
    scheme: MarkScheme, problem_results: list[dict[QuestionId, QuestionResult] | None]
) -> dict[tuple[Method[Any], QuestionId], list[Check]]:
    """The checks of methods that precompute, by method and the question they read, in the
    markings without problems."""
    precomputing_checks: dict[tuple[Method[Any], QuestionId], list[Check]] = {}
    for marking, marking_problem_results in zip(scheme.markings, problem_results, strict=True):
        if marking_problem_results is not None:
            continue
        for check, question in checks_in(marking, marking.question):
            if question is not None and type(check.method).precompute is not Method.precompute:
                precomputing_checks.setdefault((check.method, question), []).append(check)
    return precomputing_checks


def _problem_results(
    marking: Check, questions: QuestionSet
) -> dict[QuestionId, QuestionResult] | None:
    """The unmarked results of a marking with problems, which say what they are; None if it has
    none."""
    problems = marking.problems(questions)
    if not problems:
        return None
    try:
        keys = marking.questions_marked()
    except InvalidError, TemplateError:
        keys = {marking.question} if marking.question else set()
    feedback = " ".join(map(str, problems))
    return {
        key: _unmarked_result(questions[key], feedback, marking.method.label)
        for key in keys & questions.keys()
    }


def _questions_read(marking: Check) -> tuple[QuestionId, ...] | None:
    """The questions whose answers a marking reads; None if it can't say."""
    try:
        return tuple(sorted(marking.questions_read()))
    except InvalidError, TemplateError:  # a generated body that marking reports
        return None


def _marking_results(marking: Check, context: Context) -> dict[QuestionId, QuestionResult]:
    label = marking.method.label
    keys = marking.questions_marked() & context.questions.keys()
    try:
        outcomes = marking.mark(context)
    except GradeFlowError as error:  # such as code that failed on this answer
        return {
            key: _unmarked_result(context.questions[key], _unmarked_feedback(str(error)), label)
            for key in keys
        }
    except Exception:  # a bug, which shouldn't stop the other answers being marked
        _log.exception("Marking with %s failed", marking.uses)
        return {
            key: _unmarked_result(context.questions[key], _unmarked_feedback(), label)
            for key in keys
        }
    return {
        key: _result(outcomes.get(key) or Outcome(0.0), context.questions[key], label)
        for key in keys
    }


def _unmarked_feedback(reason: str = "") -> str:
    return f"Couldn't mark this answer: {reason}" if reason else "Couldn't mark this answer."


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
