from dataclasses import dataclass, replace

from ..questions.base import Question
from ..questions.question_set import QuestionSet, value_of
from ..questions.reading import write_answer
from ..questions.types import AnswerShape, QuestionId, QuestionType


@dataclass(frozen=True)
class Position:
    """Where a check is: the question and value it reads, and what holds it."""

    questions: QuestionSet
    question: QuestionId | None = None
    value: int | None = None
    in_list: bool = False
    """In another check's list, where checks have names, weights and conditions."""
    in_group: bool = False
    """In a marking of a question group, where checks choose their questions."""
    value_fixed: bool = False
    """At the value its place gives it, where checks don't choose their values."""
    one_answer: bool = False
    """Reading one of the answers its question reads each answer as."""

    @property
    def question_model(self) -> Question | None:
        """The question it reads, or one reading answers as the value it reads is read."""
        question = self.questions.get(self.question) if self.question else None
        if question is not None and self.value is not None:
            return question.value_question(self.value)
        if question is not None and self.one_answer:
            return question.single()
        return question

    @property
    def answer_shape(self) -> AnswerShape:
        """Whether what it reads is one answer or several possible answers."""
        question = self.question_model
        return question.answer_shape if question else "single"

    @property
    def answer_type(self) -> QuestionType | None:
        """The type of what the check reads: a value's type when it reads one value."""
        question = self.question_model
        return question.type if question else None

    def answer_value(self, answer: str) -> str:
        """What a check here reads of ``answer``, an answer to its question, written as the
        question writes it: the whole answer, or the value it reads; ``""`` if blank or
        unreadable."""
        if self.question is None or (question := self.questions.get(self.question)) is None:
            return ""
        try:
            read = value_of(question.read(answer), self.value)
        except ValueError:
            return ""
        return write_answer(question.first(read))

    def at(self, question: QuestionId | None, value: int | None) -> Position:
        """The position of what a check naming ``question`` and ``value`` reads."""
        if question is not None and question != self.question:
            return Position(self.questions, question, value, self.in_list, self.in_group)
        return replace(self, value=value or self.value)

    def inside(self, question: QuestionId | None, value: int | None, *, in_list: bool) -> Position:
        """The position of the checks in the settings of a check naming ``question`` and
        ``value``, before their method places them."""
        return replace(self.at(question, value), in_list=in_list, value_fixed=False)
