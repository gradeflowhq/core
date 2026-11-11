from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from ..questions.base import Question
from ..questions.question_set import QuestionSet, value_of
from ..questions.reading import write_answer
from ..questions.types import QuestionId, QuestionType
from .input_sources import (
    AnswerSource,
    AnswerValueSource,
    InputSource,
    OtherAnswerSource,
    StepOutputSource,
)
from .templates import unknown_step_message

MARKS_AS_ANSWER_MESSAGE = "Marks can't be read as an answer."
type OutputQuestions = Mapping[str, Question | None]
"""By earlier step ID, the question that reads its output; None for a check."""


@dataclass(frozen=True)
class Position:
    """Where a step is: the question and input it reads, and what holds it."""

    questions: QuestionSet
    question: QuestionId | None = None
    source: InputSource = field(default_factory=AnswerSource)
    """What its input names: the answer as it is, unless another."""
    input_question: Question | None = None
    """The question that reads its input, when that isn't the answer as it is."""
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

    @property
    def value(self) -> int | None:
        """Which value of the answer its input reads."""
        return self.source.value_number if isinstance(self.source, AnswerValueSource) else None

    @property
    def input_kind(self) -> Literal["value", "output"] | None:
        """What its input is, when it isn't a question's answer."""
        match self.source:
            case AnswerValueSource():
                return "value"
            case StepOutputSource():
                return "output"
        return None

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
            return replace(self, question=question, source=AnswerSource(), input_question=None)
        return self

    def reading(
        self, source: InputSource, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        """This position once it reads ``source``, and why it can't, if it can't."""
        match source:
            case AnswerSource():
                return self, None
            case AnswerValueSource(value_number=value_number):
                question = self.questions.get(self.question or "")
                if question is None or not question.value_count:
                    return self, "Only a question with several values has values to read."
                value_question = question.value_question(value_number)
                if value_question is None:
                    return self, f"The answer has {question.value_count} values."
                return replace(self, source=source, input_question=value_question), None
            case OtherAnswerSource(question_id=question_id):
                other_question = self.questions.get(question_id)
                if other_question is None:
                    return self, f"{question_id} isn't one of the questions."
                return replace(self, source=source, input_question=other_question), None
            case StepOutputSource(step_id=step_id):
                if step_id not in output_questions:
                    return self, unknown_step_message(step_id)
                output_question = output_questions[step_id]
                if output_question is None:
                    return self, MARKS_AS_ANSWER_MESSAGE
                return replace(self, source=source, input_question=output_question), None
