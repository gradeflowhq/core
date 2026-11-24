"""What marking one student needs."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..models.model_service import ModelEndpoint
from ..questions.base import Question
from ..questions.question_set import Answers, QuestionSet, value_of
from ..questions.reading import write_answer
from ..questions.types import Answer, QuestionId, Unreadable
from .outcome import Span, StepResult, Transformed, TransformResult


@dataclass(frozen=True)
class Progress:
    """How far a step of marking has got, such as 12 of 30 requests to a language model."""

    completed: int
    total: int
    step: str


@dataclass(frozen=True)
class StepInput:
    """What a step reads: an answer with the question that reads it."""

    question_id: QuestionId
    """The question whose marks it decides."""
    answer: Answer | Unreadable
    question: Question
    written: str
    """The answer as the student wrote it, or as a step that gave it writes it."""
    answer_span: Span | None
    """Where its text is in the answer to ``question_id``; None if it isn't part of that answer,
    such as another question's answer."""

    def mapped_to_answer(self, result: TransformResult) -> TransformResult:
        """``result``, what a transform gives of this input, with its spans mapped from this input
        to the answer to ``question_id``, or dropped if this input isn't part of that answer.
        Raises ``ValueError`` if a span ends past this input."""
        if isinstance(result, Unreadable):
            return result
        if self.answer_span is None:
            return Transformed(result.answer)
        spans = tuple(self.answer_span.subspan(span) for span in result.spans)
        return Transformed(result.answer, spans)


@dataclass(frozen=True)
class Context:
    """What marking one student needs: their answers, the questions, the results of earlier steps
    and what the step being run reads."""

    answers: Answers
    questions: QuestionSet
    written_answers: Mapping[QuestionId, str]
    earlier_results: Mapping[str, StepResult] = field(default_factory=dict)
    input: StepInput | None = None
    """What the step being run reads, when that is one answer."""
    method_settings: Mapping[str, Any] = field(default_factory=dict)
    """The settings of the method that wrote the step, if one did."""
    model_endpoint: ModelEndpoint | None = None
    report_progress: Callable[[Progress], None] = lambda _: None

    def answer_to(self, question_id: QuestionId) -> StepInput:
        """The student's answer to ``question_id``."""
        answer = self.answers.get(question_id)
        return StepInput(
            question_id,
            answer,
            self.questions[question_id],
            self.written_answers.get(question_id, ""),
            _full_span(answer),
        )

    def input_from(
        self,
        question_id: QuestionId,
        answer: Answer | Unreadable,
        question: Question,
        answer_span: Span | None,
    ) -> StepInput:
        """``answer``, read by ``question``, as what decides the marks of ``question_id``, at
        ``answer_span`` of that question's answer."""
        max_marks = self.questions[question_id].max_marks
        return StepInput(
            question_id,
            answer,
            question
            if question.max_marks == max_marks
            else question.model_copy(update={"max_marks": max_marks}),
            _answer_text(answer),
            answer_span,
        )

    def value(self, question_id: QuestionId, value_number: int) -> StepInput:
        """Value ``value_number`` of the student's answer to ``question_id``."""
        question = self.questions[question_id].value_question(value_number)
        assert question is not None  # problems refuse a value the answers don't have
        value = value_of(self.answers.get(question_id), value_number)
        return self.input_from(question_id, value, question, _full_span(value, value_number))

    def text_at(self, question_id: QuestionId, span: Span) -> str:
        """The text at ``span`` in the student's answer to ``question_id``."""
        answer = value_of(self.answers.get(question_id), span.value_number)
        return _answer_text(answer)[span.start : span.end]

    def written_answer(self, question_id: QuestionId) -> str:
        """The answer to ``question_id`` as feedback shows it: what the step reads, if it reads
        another input for it."""
        if self.input is not None and self.input.question_id == question_id:
            return self.input.written
        return self.written_answers.get(question_id, "")


def _answer_text(answer: Answer | Unreadable) -> str:
    """``answer`` as a step that gives it writes it; empty if it is unreadable."""
    return "" if isinstance(answer, Unreadable) else write_answer(answer)


def _full_span(answer: Answer | Unreadable, value_number: int | None = None) -> Span:
    """The span of all of ``answer``, as value ``value_number`` of its answer if given."""
    return Span(0, len(_answer_text(answer)), value_number)
