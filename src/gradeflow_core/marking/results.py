from dataclasses import dataclass

from natsort import natsorted
from pydantic import BaseModel, Field, computed_field

from ..common.text import format_rounded
from ..methods.outcome import FULL_MARKS
from ..questions.submissions import AnswerSheet, Submission
from ..questions.types import QuestionId


class QuestionResult(BaseModel):
    """What marking gave one answer: a proportion of the question's maximum marks."""

    proportion: float = Field(
        ge=0, le=1, description="The proportion of the marks earned, from 0 to 1"
    )
    max_marks: float = Field(ge=0, description="The question's maximum marks")
    feedback: str = ""
    method: str = Field(default="", description="The label of the method that marked it")
    marked: bool = Field(default=True, description="False when it needs a person to mark it")
    models: tuple[str, ...] = Field(
        default=(), description="The language models that decided it, if any"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def marks(self) -> float:
        return self.proportion * self.max_marks

    @property
    def correct(self) -> bool:
        return self.proportion >= FULL_MARKS


@dataclass(frozen=True)
class Report:
    """What a student got for one question: its marks out of its maximum, and any feedback."""

    question: QuestionId
    marks: float
    max_marks: float
    feedback: str

    @property
    def heading(self) -> str:
        """The question and its marks, such as ``Q3 [1/3]``."""
        return f"{self.question} [{format_rounded(self.marks)}/{format_rounded(self.max_marks)}]"

    def __str__(self) -> str:
        return f"{self.heading}\n{self.feedback}" if self.feedback else self.heading


class MarkedSheet(AnswerSheet):
    """An answer sheet with what marking gave it: a result for each question it was marked for."""

    results: dict[QuestionId, QuestionResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_marks(self) -> float:
        return sum(result.marks for result in self.results.values())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_marks(self) -> float:
        return sum(result.max_marks for result in self.results.values())

    def report(self) -> list[Report]:
        """The marking report: the marks and feedback, question by question, for a gradebook
        comment."""
        return [
            Report(key, result.marks, result.max_marks, result.feedback.strip())
            for key, result in natsorted(self.results.items(), key=lambda item: item[0])
        ]


class MarkedSubmission(Submission, MarkedSheet):
    """A student's marked answer sheet."""
