from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from natsort import natsorted
from pydantic import Field, RootModel, field_validator

from ..common.documents import dump_yaml, load_yaml
from ..common.problems import as_invalid
from . import AnyQuestion
from .base import Question
from .inference import infer_question
from .submissions import AnswerSheet
from .types import Answer, QuestionId, Unreadable, Value

type Answers = Mapping[QuestionId, Answer | Unreadable]
"""A student's answers as the questions read them."""


def read_answer(answers: Answers, question: QuestionId, value: int | None) -> Answer | Unreadable:
    """A student's answer to ``question``, or the value of it a check reads; None when blank."""
    return value_of(answers.get(question), value)


def value_of[A: Answer | Unreadable](answer: A, value: int | None) -> A | Value:
    """``answer``, or its value ``value``, from 1, when it has values; None when it has fewer."""
    if value is not None and isinstance(answer, list):
        return answer[value - 1] if value <= len(answer) else None
    return answer


class QuestionSet(RootModel[dict[QuestionId, AnyQuestion]], Mapping[QuestionId, Question]):
    """Questions by ID, in natural ID order: Q2 before Q10."""

    root: dict[QuestionId, AnyQuestion] = Field(default_factory=dict)

    @field_validator("root")
    @classmethod
    def _natural_order(
        cls, questions: dict[QuestionId, AnyQuestion]
    ) -> dict[QuestionId, AnyQuestion]:
        return {key: questions[key] for key in natsorted(questions)}

    @classmethod
    def load(cls, data: object) -> QuestionSet:
        """Questions by ID as YAML or JSON holds them. Raises ``InvalidError`` if one isn't a
        question."""
        with as_invalid(data):
            return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, text: str) -> QuestionSet:
        """The questions YAML ``text`` holds, as ``to_yaml`` writes them. Raises ``InvalidError``
        if it isn't YAML or one isn't a question."""
        return cls.load(load_yaml(text))

    def to_yaml(self) -> str:
        """The questions as YAML text, as ``from_yaml`` reads them."""
        return dump_yaml(self.dump())

    def dump(self) -> dict[QuestionId, dict[str, Any]]:
        """The questions as YAML or JSON holds them, as ``load`` reads them: each with its type,
        without what it leaves at the defaults."""
        return {key: question.dump() for key, question in self.items()}

    def __getitem__(self, key: QuestionId) -> Question:
        return self.root[key]

    def __iter__(self) -> Iterator[QuestionId]:  # type: ignore[override]  # keys, as a mapping
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def read(self, sheet: AnswerSheet) -> Answers:
        """Each answer as its question reads it; answers to questions outside the set are left
        out."""
        answers: dict[QuestionId, Answer | Unreadable] = {}
        for key, text in sheet.answers.items():
            if (question := self.get(key)) is not None:
                try:
                    answers[key] = question.read(text)
                except ValueError as error:
                    answers[key] = Unreadable(text, str(error))
        return answers

    def updated_from(self, sheets: Sequence[AnswerSheet]) -> QuestionSet:
        """Only answered questions, with existing ones changed to accept their answers and new ones
        worked out, out of the highest marks a sheet was given for them."""
        questions: dict[QuestionId, Question] = {}
        for key in {key for sheet in sheets for key in sheet.answers}:
            answers = [sheet.answers[key] for sheet in sheets if key in sheet.answers]
            if (question := self.get(key)) is not None:
                questions[key] = question.accepting(answers)
                continue
            new = infer_question(answers)
            highest = max((s.marks[key] for s in sheets if key in s.marks), default=0)
            questions[key] = new.model_copy(update={"max_marks": highest}) if highest > 0 else new
        return type(self)(questions)
