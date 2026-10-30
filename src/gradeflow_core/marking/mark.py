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
    blocked = [_blocked(marking, questions) for marking in scheme.markings]
    reads = [_reads(marking) for marking in scheme.markings]
    caches: list[dict[tuple[str | None, ...], dict[QuestionId, QuestionResult]]] = [
        {} for _ in scheme.markings
    ]
    groups = _precomputing(scheme, blocked)
    precomputed: dict[str, dict[QuestionId, dict[Any, Any]]] = {}
    asked: set[tuple[str, QuestionId, str]] = set()
    # Without methods that precompute, sheets are read one at a time, as each is marked.
    for chunk in batched(sheets, chunk_size if groups else 1, strict=False):
        read = [questions.read(raw) for raw in chunk]
        for (method, question), checks in groups.items():
            texts = {
                raw.answers[question].strip(): None
                for raw, answers in zip(chunk, read, strict=True)
                if (answer := read_answer(answers, question, None)) is not None
                and not isinstance(answer, Unreadable)
                and (method.ref, question, raw.answers[question].strip()) not in asked
            }
            if not texts:
                continue
            asked.update((method.ref, question, text) for text in texts)
            found = method.precompute(
                checks,
                list(texts),
                questions[question],
                model_endpoint,
                on_progress or (lambda _: None),
            )
            precomputed.setdefault(method.ref, {}).setdefault(question, {}).update(found)
        for raw, answers in zip(chunk, read, strict=True):
            results: dict[QuestionId, QuestionResult] = {}
            for marking, unmarked, keys, cache in zip(
                scheme.markings, blocked, reads, caches, strict=True
            ):
                if unmarked is not None:
                    marked = unmarked
                elif keys is None:
                    context = Context(
                        answers, questions, written=raw.answers, precomputed=precomputed
                    )
                    marked = _results(marking, context)
                else:
                    identity = tuple(raw.answers.get(key) for key in keys)
                    if identity not in cache:
                        seen = {key: answers[key] for key in keys if key in answers}
                        written = {key: raw.answers[key] for key in keys if key in raw.answers}
                        context = Context(seen, questions, written=written, precomputed=precomputed)
                        cache[identity] = _results(marking, context)
                    marked = cache[identity]
                results = {key: result.model_copy() for key, result in marked.items()} | results
            results = _given(raw, questions) | results
            kind = MarkedSubmission if isinstance(raw, Submission) else MarkedSheet
            yield kind(
                **raw.model_dump(include=set(kind.model_fields)),
                results={key: results[key] for key in questions if key in results},
            )


def _precomputing(
    scheme: MarkScheme, blocked: list[dict[QuestionId, QuestionResult] | None]
) -> dict[tuple[Method[Any], QuestionId], list[Check]]:
    """The checks of methods that precompute, by method and the question they read, in the
    markings without problems."""
    groups: dict[tuple[Method[Any], QuestionId], list[Check]] = {}
    for marking, unmarked in zip(scheme.markings, blocked, strict=True):
        if unmarked is not None:
            continue
        for check, question in checks_in(marking, marking.question):
            if question is not None and type(check.method).precompute is not Method.precompute:
                groups.setdefault((check.method, question), []).append(check)
    return groups


def _blocked(marking: Check, questions: QuestionSet) -> dict[QuestionId, QuestionResult] | None:
    """The unmarked results of a marking with problems, which say what they are; None if it has
    none."""
    found = marking.problems(questions)
    if not found:
        return None
    try:
        keys = marking.questions_marked()
    except InvalidError, TemplateError:
        keys = {marking.question} if marking.question else set()
    feedback = " ".join(map(str, found))
    return {
        key: _unmarked(questions[key], feedback, marking.method.label)
        for key in keys & questions.keys()
    }


def _reads(marking: Check) -> tuple[QuestionId, ...] | None:
    """The questions whose answers a marking reads; None if it can't say."""
    try:
        return tuple(sorted(marking.questions_read()))
    except InvalidError, TemplateError:  # a generated body that marking reports
        return None


def _results(marking: Check, context: Context) -> dict[QuestionId, QuestionResult]:
    label = marking.method.label
    keys = marking.questions_marked() & context.questions.keys()
    try:
        outcomes = marking.mark(context)
    except GradeFlowError as error:  # such as code that failed on this answer
        return {key: _unmarked(context.questions[key], _couldnt(str(error)), label) for key in keys}
    except Exception:  # a bug, which shouldn't stop the other answers being marked
        _log.exception("Marking with %s failed", marking.uses)
        return {key: _unmarked(context.questions[key], _couldnt(), label) for key in keys}
    return {
        key: _result(outcomes.get(key) or Outcome(0.0), context.questions[key], label)
        for key in keys
    }


def _couldnt(reason: str = "") -> str:
    return f"Couldn't mark this answer: {reason}" if reason else "Couldn't mark this answer."


def _result(outcome: Outcome, question: Question, label: str) -> QuestionResult:
    return QuestionResult(
        proportion=clamped(outcome.proportion),
        max_marks=question.max_marks,
        feedback=outcome.feedback,
        method=outcome.method or label,
        marked=outcome.marked,
        models=outcome.all_models(),
    )


def _unmarked(question: Question, feedback: str, label: str) -> QuestionResult:
    return _result(Outcome(0.0, feedback, marked=False), question, label)


def _given(raw: AnswerSheet, questions: QuestionSet) -> dict[QuestionId, QuestionResult]:
    """The marks the sheet was given, as results with no method, within each question's."""
    return {
        key: QuestionResult(
            proportion=clamped(marks / question.max_marks) if question.max_marks else 0.0,
            max_marks=question.max_marks,
        )
        for key, marks in raw.marks.items()
        if (question := questions.get(key)) is not None
    }
