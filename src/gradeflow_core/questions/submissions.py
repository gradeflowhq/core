from pydantic import BaseModel, Field

from .types import QuestionId

StudentId = str


class AnswerSheet(BaseModel):
    """Answers as written, by question, and any marks they were already given."""

    answers: dict[QuestionId, str]
    marks: dict[QuestionId, float] = Field(
        default_factory=dict,
        description="Marks already given, such as by an exam system, kept where nothing marks",
    )


class Submission(AnswerSheet):
    """A student's answer sheet."""

    student_id: StudentId
