from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ..common.text import format_marks
from ..methods.outcome import FULL_MARKS_PROPORTION, Detail, Effect
from ..questions.submissions import AnswerSheet, Submission
from ..questions.types import QuestionId


class CheckResult(BaseModel):
    """How a check of a marking marked an answer, and how its own checks did."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="What it checks, in a line of markdown")
    effect: Effect = Field(description="What it did to the marks")
    marks: float | None = Field(
        default=None, description="The marks it gave, or its penalty, where its checks add up"
    )
    details: tuple[Detail, ...] = ()
    children: tuple[CheckResult, ...] = ()


class QuestionResult(BaseModel):
    """What marking gave one answer: a proportion of the question's maximum marks."""

    proportion: float = Field(
        ge=0, le=1, description="The proportion of the marks earned, from 0 to 1"
    )
    max_marks: float = Field(ge=0, description="The question's maximum marks")
    feedback: str = ""
    method_label: str = Field(default="", description="The label of the method that marked it")
    is_marked: bool = Field(default=True, description="False when it needs a person to mark it")
    model_names: tuple[str, ...] = Field(
        default=(), description="The language models that decided it, if any"
    )
    breakdown: CheckResult | None = Field(
        default=None, description="How each check marked it, when the marking has several"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def marks(self) -> float:
        return self.proportion * self.max_marks

    @property
    def correct(self) -> bool:
        return self.proportion >= FULL_MARKS_PROPORTION


@dataclass(frozen=True)
class ReportLine:
    """What a student got for one question: its marks out of its maximum, and any feedback."""

    heading: str
    """The question and its marks, such as ``Q3 [1/3]``."""
    feedback: str

    def __str__(self) -> str:
        return f"{self.heading}\n{self.feedback}" if self.feedback else self.heading


class MarkedSheet(AnswerSheet):
    """An answer sheet with what marking gave it: a result for each question it was marked for."""

    results: dict[QuestionId, QuestionResult]
    max_marks: float = Field(ge=0, description="The maximum marks of all the questions")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_marks(self) -> float:
        return sum(result.marks for result in self.results.values())

    def report(self) -> list[ReportLine]:
        """The marking report: the marks and feedback, question by question, for a gradebook
        comment."""
        return [
            ReportLine(
                f"{question_id} [{format_marks(result.marks)}/{format_marks(result.max_marks)}]",
                result.feedback.strip(),
            )
            for question_id, result in self.results.items()
        ]


class MarkedSubmission(Submission, MarkedSheet):
    """A student's marked answer sheet."""
