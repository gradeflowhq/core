"""What a step's ``input`` names, in place of the answer as it is."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jinja2 import nodes

from ..questions.base import Question
from ..questions.types import QuestionId
from .outcome import Outcome, Skipped
from .position import Position
from .templates import expression_node

if TYPE_CHECKING:
    from .step import Context

ANSWER_INPUT = "{{ answer }}"
"""What a step without an ``input`` reads."""

type OutputQuestions = Mapping[str, Question | None]
"""By earlier step ID, the question that reads its output; None for a check."""


@dataclass(frozen=True)
class ResolvedInput:
    """A step's input read for one student. ``short_circuit_outcome`` replaces running the step."""

    context: Context
    short_circuit_outcome: Outcome | None = None


class InputSource(ABC):
    @abstractmethod
    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        """``position`` once it reads this, and the error if it can't."""

    @abstractmethod
    def resolve(self, context: Context, key: QuestionId | None) -> ResolvedInput: ...

    def question_key(self, question: QuestionId | None) -> QuestionId | None:
        """The answer a step reading this at ``question`` replaces, and marks."""
        return question

    def questions_read(self) -> set[QuestionId]:
        return set()

    def input_label(self, label_of_step: Callable[[str], str]) -> str | None:
        """What it is, in words, if it isn't the answer as it is."""
        return None

    def default_step_label(self) -> str | None:
        """What a step reading it is called, if that says more than its method."""
        return None

    def input_question(
        self, output_questions: Mapping[str, Question], source_question: Question
    ) -> Question:
        """The question a step reading this reads."""
        return source_question


@dataclass(frozen=True)
class AnswerSource(InputSource):
    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        return position, None

    def resolve(self, context: Context, key: QuestionId | None) -> ResolvedInput:
        return ResolvedInput(context)


@dataclass(frozen=True)
class AnswerValueSource(InputSource):
    index: int
    """From 0, as a template counts."""

    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        source_question = position.questions.get(position.question or "")
        if source_question is None or not source_question.value_count:
            return position, "Only a question with several values has values to read."
        value_question = source_question.value_question(self.index + 1)
        if value_question is None:
            return position, f"The answer has {source_question.value_count} values."
        return position.with_input(value_question, "value", self.index + 1), None

    def resolve(self, context: Context, key: QuestionId | None) -> ResolvedInput:
        assert key is not None  # problems refuse values with no question to read
        return ResolvedInput(context.with_value_as_answer(key, self.index + 1))

    def input_label(self, label_of_step: Callable[[str], str]) -> str | None:
        return f"value {self.index + 1}"

    def default_step_label(self) -> str | None:
        return f"Value {self.index + 1}"

    def input_question(
        self, output_questions: Mapping[str, Question], source_question: Question
    ) -> Question:
        return source_question.value_question(self.index + 1) or source_question


@dataclass(frozen=True)
class OtherAnswerSource(InputSource):
    question: QuestionId

    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        other_question = position.questions.get(self.question)
        if other_question is None:
            return position, f"{self.question} isn't one of the questions."
        return position.with_input(other_question), None

    def resolve(self, context: Context, key: QuestionId | None) -> ResolvedInput:
        assert key is not None  # ``question_key`` falls back to this source's own question
        answer = context.answers.get(self.question)
        return ResolvedInput(context.with_answer(key, answer, context.questions[self.question]))

    def question_key(self, question: QuestionId | None) -> QuestionId | None:
        return question or self.question

    def questions_read(self) -> set[QuestionId]:
        return {self.question}

    def input_label(self, label_of_step: Callable[[str], str]) -> str | None:
        return str(self.question)


@dataclass(frozen=True)
class StepOutputSource(InputSource):
    step: str

    def input_position(
        self, position: Position, output_questions: OutputQuestions
    ) -> tuple[Position, str | None]:
        if self.step not in output_questions:
            return position, f"No step before this one is called {self.step}."
        output = output_questions[self.step]
        if output is None:
            return position, "Marks can't be read as an answer."
        return position.with_input(output, "output"), None

    def resolve(self, context: Context, key: QuestionId | None) -> ResolvedInput:
        result = context.earlier_results.get(self.step)
        if result is None or isinstance(answer := result.transform_result, Skipped):
            return ResolvedInput(context, Outcome(0.0, skipped=True))
        assert key is not None  # problems refuse a step reading an output with no question
        assert result.question is not None  # problems refuse reading a check's marks
        return ResolvedInput(context.with_answer(key, answer, result.question))

    def input_label(self, label_of_step: Callable[[str], str]) -> str | None:
        return f"the output of {label_of_step(self.step)}"

    def input_question(
        self, output_questions: Mapping[str, Question], source_question: Question
    ) -> Question:
        return output_questions.get(self.step, source_question)


_INPUT_RULE = "Write an input as one of {{ steps.name.output }}, {{ values[0] }}, {{ answers.Q1 }}."


def input_source(input_text: object) -> InputSource:
    """The source a step's ``input`` names. Raises ``ValueError`` if it names none."""
    match expression_node(ANSWER_INPUT if input_text is None else str(input_text)):
        case nodes.Name(name="answer"):
            return AnswerSource()
        case nodes.Getitem(node=nodes.Name(name="values"), arg=nodes.Const(value=int(index))):
            return AnswerValueSource(index)
        case (
            nodes.Getattr(node=nodes.Name(name="answers"), attr=key)
            | nodes.Getitem(node=nodes.Name(name="answers"), arg=nodes.Const(value=str(key)))
        ):
            return OtherAnswerSource(key)
        case nodes.Getattr(
            node=nodes.Getattr(node=nodes.Name(name="steps"), attr=step), attr="output"
        ):
            return StepOutputSource(step)
    raise ValueError(_INPUT_RULE)
