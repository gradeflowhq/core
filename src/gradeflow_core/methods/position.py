from dataclasses import dataclass, replace
from typing import Literal

from ..questions.base import Question
from ..questions.question_set import QuestionSet, value_of
from ..questions.reading import write_answer
from ..questions.types import QuestionId, QuestionType


@dataclass(frozen=True)
class Position:
    """Where a step is: the question and input it reads, and what holds it."""

    questions: QuestionSet
    question: QuestionId | None = None
    value: int | None = None
    """Which value of the answer its input reads."""
    input_question: Question | None = None
    """The question that reads its input, when that isn't the answer as it is."""
    input_kind: Literal["value", "output"] | None = None
    """What its input is, when it isn't another question's answer."""
    is_in_list: bool = False
    """In another check's list, where checks have names, weights and conditions."""
    is_in_group: bool = False
    """In a marking of a question group, where checks choose their questions."""
    is_in_workflow: bool = False
    """In a workflow's steps, where transforms can be and inputs can be chosen."""

    @property
    def question_model(self) -> Question | None:
        """The question that reads its input."""
        if self.input_question is not None:
            return self.input_question
        return self.questions.get(self.question) if self.question else None

    @property
    def question_type(self) -> QuestionType | None:
        """The type of what the step reads."""
        question = self.question_model
        return question.type if question else None

    def answer_value(self, answer: str) -> str:
        """The part of ``answer`` a check here reads, as written; ``""`` if blank or unreadable."""
        if self.question is None or (question := self.questions.get(self.question)) is None:
            return ""
        try:
            value = value_of(question.read(answer), self.value)
        except ValueError:
            return ""
        return write_answer(value, question.reading)

    def for_question(self, question: QuestionId | None) -> Position:
        """The position of what a step naming ``question`` reads."""
        if question is not None and question != self.question:
            return replace(
                self, question=question, value=None, input_question=None, input_kind=None
            )
        return self

    def with_input(
        self,
        input_question: Question,
        input_kind: Literal["value", "output"] | None = None,
        value: int | None = None,
    ) -> Position:
        """This position, reading what ``input_question`` reads."""
        return replace(self, input_question=input_question, input_kind=input_kind, value=value)
