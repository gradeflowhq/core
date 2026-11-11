"""What marking one student needs."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ..models.model_service import ModelEndpoint
from ..questions.base import Question
from ..questions.question_set import Answers, QuestionSet, value_of
from ..questions.reading import write_answer
from ..questions.types import Answer, QuestionId, Unreadable
from .outcome import StepResult


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
    model_endpoint: ModelEndpoint | None = None
    report_progress: Callable[[Progress], None] = lambda _: None

    def answer_to(self, question_id: QuestionId) -> StepInput:
        """The student's answer to ``question_id``."""
        return StepInput(
            question_id,
            self.answers.get(question_id),
            self.questions[question_id],
            self.written_answers.get(question_id, ""),
        )

    def input_from(
        self, question_id: QuestionId, answer: Answer | Unreadable, question: Question
    ) -> StepInput:
        """``answer``, read by ``question``, as what decides the marks of ``question_id``."""
        max_marks = self.questions[question_id].max_marks
        return StepInput(
            question_id,
            answer,
            question
            if question.max_marks == max_marks
            else question.model_copy(update={"max_marks": max_marks}),
            "" if isinstance(answer, Unreadable) else write_answer(answer),
        )

    def value(self, question_id: QuestionId, value_number: int) -> StepInput:
        """Value ``value_number`` of the student's answer to ``question_id``."""
        question = self.questions[question_id].value_question(value_number)
        assert question is not None  # problems refuse a value the answers don't have
        return self.input_from(
            question_id, value_of(self.answers.get(question_id), value_number), question
        )

    def written_answer(self, question_id: QuestionId) -> str:
        """The answer to ``question_id`` as feedback shows it: what the step reads, if it reads
        another input for it."""
        if self.input is not None and self.input.question_id == question_id:
            return self.input.written
        return self.written_answers.get(question_id, "")
