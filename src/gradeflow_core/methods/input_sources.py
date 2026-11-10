"""What a step's ``input`` names, in place of the answer as it is."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jinja2 import nodes

from ..questions.base import Question
from ..questions.types import QuestionId
from .outcome import SkippedStep
from .position import Position
from .templates import expression_node, unknown_step_message

if TYPE_CHECKING:
    from .step import Context

ANSWER_INPUT = "{{ answer }}"
"""What a step without an ``input`` reads."""

MARKS_AS_ANSWER_MESSAGE = "Marks can't be read as an answer."
type OutputQuestions = Mapping[str, Question | None]
"""By earlier step ID, the question that reads its output; None for a check."""


class InputSource(ABC):
    @property
    @abstractmethod
    def input_text(self) -> str:
        """How a step's ``input`` names it."""

    @abstractmethod
    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        """``position`` once it reads this, and the error if it can't."""

    @abstractmethod
    def resolved_context(self, context: Context, question_id: QuestionId | None) -> Context | None:
        """``context`` with this read as the answer to ``question_id``; None when the step it reads
        didn't run."""

    def marked_question_id(self, question_id: QuestionId | None) -> QuestionId | None:
        """The answer a step reading this at ``question_id`` replaces, and marks."""
        return question_id

    def questions_read(self) -> set[QuestionId]:
        return set()

    def input_label(self, label_of_step: Callable[[str], str]) -> str | None:
        """What it is, in words, if it isn't the answer as it is."""
        return None

    def default_step_label(self) -> str | None:
        """What a step reading it is called, if that says more than its method."""
        return None


@dataclass(frozen=True)
class AnswerSource(InputSource):
    @property
    def input_text(self) -> str:
        return ANSWER_INPUT

    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        return position, None

    def resolved_context(self, context: Context, question_id: QuestionId | None) -> Context | None:
        return context


@dataclass(frozen=True)
class AnswerValueSource(InputSource):
    value_number: int
    """From 1."""

    @property
    def input_text(self) -> str:
        return f"{{{{ values[{self.value_number - 1}] }}}}"

    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        source_question = position.questions.get(position.question or "")
        if source_question is None or not source_question.value_count:
            return position, "Only a question with several values has values to read."
        value_question = source_question.value_question(self.value_number)
        if value_question is None:
            return position, f"The answer has {source_question.value_count} values."
        return position.with_input(value_question, "value", self.value_number), None

    def resolved_context(self, context: Context, question_id: QuestionId | None) -> Context | None:
        assert question_id is not None  # problems refuse values with no question to read
        return context.with_value_as_answer(question_id, self.value_number)

    def input_label(self, label_of_step: Callable[[str], str]) -> str | None:
        return f"value {self.value_number}"

    def default_step_label(self) -> str | None:
        return f"Value {self.value_number}"


@dataclass(frozen=True)
class OtherAnswerSource(InputSource):
    question_id: QuestionId

    @property
    def input_text(self) -> str:
        if self.question_id.isidentifier():
            return f"{{{{ answers.{self.question_id} }}}}"
        return f"{{{{ answers[{self.question_id!r}] }}}}"

    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        other_question = position.questions.get(self.question_id)
        if other_question is None:
            return position, f"{self.question_id} isn't one of the questions."
        return position.with_input(other_question), None

    def resolved_context(self, context: Context, question_id: QuestionId | None) -> Context | None:
        # ``marked_question_id`` falls back to this source's own question
        assert question_id is not None
        answer = context.answers.get(self.question_id)
        return context.with_answer(question_id, answer, context.questions[self.question_id])

    def marked_question_id(self, question_id: QuestionId | None) -> QuestionId | None:
        return question_id or self.question_id

    def questions_read(self) -> set[QuestionId]:
        return {self.question_id}

    def input_label(self, label_of_step: Callable[[str], str]) -> str | None:
        return str(self.question_id)


@dataclass(frozen=True)
class StepOutputSource(InputSource):
    step_id: str

    @property
    def input_text(self) -> str:
        return f"{{{{ steps.{self.step_id}.output }}}}"

    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        if self.step_id not in output_questions:
            return position, unknown_step_message(self.step_id)
        output_question = output_questions[self.step_id]
        if output_question is None:
            return position, MARKS_AS_ANSWER_MESSAGE
        return position.with_input(output_question, "output"), None

    def resolved_context(self, context: Context, question_id: QuestionId | None) -> Context | None:
        result = context.earlier_results.get(self.step_id)
        if result is None or isinstance(answer := result.transform_result, SkippedStep):
            return None
        assert question_id is not None  # problems refuse a step reading an output with no question
        assert result.question is not None  # problems refuse reading a check's marks
        return context.with_answer(question_id, answer, result.question)

    def input_label(self, label_of_step: Callable[[str], str]) -> str | None:
        return f"the output of {label_of_step(self.step_id)}"


_INPUT_RULE = "Write an input as one of {{ steps.name.output }}, {{ values[0] }}, {{ answers.Q1 }}."


def input_source(input_text: object) -> InputSource:
    """The source a step's ``input`` names. Raises ``ValueError`` if it names none."""
    match expression_node(ANSWER_INPUT if input_text is None else str(input_text)):
        case nodes.Name(name="answer"):
            return AnswerSource()
        case nodes.Getitem(node=nodes.Name(name="values"), arg=nodes.Const(value=int(index))):
            return AnswerValueSource(index + 1)
        case (
            nodes.Getattr(node=nodes.Name(name="answers"), attr=question_id)
            | nodes.Getitem(
                node=nodes.Name(name="answers"), arg=nodes.Const(value=str(question_id))
            )
        ):
            return OtherAnswerSource(question_id)
        case nodes.Getattr(
            node=nodes.Getattr(node=nodes.Name(name="steps"), attr=step_id), attr="output"
        ):
            return StepOutputSource(step_id)
    raise ValueError(_INPUT_RULE)
