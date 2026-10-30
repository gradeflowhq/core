import csv
from collections.abc import Iterable
from io import StringIO

from natsort import natsorted
from pydantic import BaseModel, ConfigDict, Field

from ..common.problems import as_invalid
from ..common.text import format_rounded
from ..marking.results import MarkedSubmission
from ..questions.question_set import QuestionSet


class ResultsCsv(BaseModel):
    """What a results CSV holds: a row for each student."""

    model_config = ConfigDict(title="CSV", str_strip_whitespace=True)

    student_id_column: str = Field(
        default="student_id",
        min_length=1,
        title="Student ID column",
        description="Name of the column of student IDs.",
    )
    include_answers: bool = Field(
        default=True, title="Include answers", description="Add each student's answers."
    )
    include_marks: bool = Field(
        default=True, title="Include marks", description="Add the marks for each question."
    )
    include_feedback: bool = Field(
        default=True, title="Include feedback", description="Add the feedback for each question."
    )
    include_total: bool = Field(
        default=True,
        title="Include total marks",
        description="Add each student's total marks and the maximum marks.",
    )
    include_report: bool = Field(
        default=True,
        title="Include marking report",
        description="Add each student's marks and feedback by question, for gradebook comments.",
    )

    @classmethod
    def load(cls, data: object) -> ResultsCsv:
        """``data`` as options. Raises ``InvalidError`` if they aren't valid."""
        with as_invalid(data):
            return cls.model_validate(data)

    def write(self, questions: QuestionSet, marked: Iterable[MarkedSubmission]) -> str:
        """The marked submissions as CSV text, a row for each student in natural ID order."""
        out = StringIO()
        writer = csv.writer(out)
        writer.writerow(self._header(questions))
        for student in natsorted(marked, key=lambda student: student.student_id):
            writer.writerow(self._cells(student, questions))
        return out.getvalue()

    def _header(self, questions: QuestionSet) -> list[str]:
        header = [self.student_id_column]
        for key in questions:
            header += [key] if self.include_answers else []
            header += [f"{key} marks"] if self.include_marks else []
            header += [f"{key} feedback"] if self.include_feedback else []
        header += ["total marks", "maximum marks"] if self.include_total else []
        header += ["report"] if self.include_report else []
        return header

    def _cells(self, student: MarkedSubmission, questions: QuestionSet) -> list[str]:
        cells = [student.student_id]
        for key in questions:
            result = student.results.get(key)
            cells += [student.answers.get(key, "")] if self.include_answers else []
            if self.include_marks:
                cells.append(format_rounded(result.marks) if result else "")
            cells += [result.feedback if result else ""] if self.include_feedback else []
        if self.include_total:
            # Out of the whole assessment's marks, not only those of the questions it has results
            # for.
            cells += [
                format_rounded(student.total_marks),
                format_rounded(sum(q.max_marks for q in questions.values())),
            ]
        if self.include_report:
            cells.append("\n".join(map(str, student.report())))
        return cells
