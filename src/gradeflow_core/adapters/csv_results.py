import csv
from collections.abc import Iterable
from io import StringIO

from natsort import natsorted
from pydantic import BaseModel, ConfigDict, Field

from ..common.exceptions import GradeFlowError
from ..common.problems import invalid_errors
from ..common.text import format_marks, quoted
from ..marking.results import Marked
from ..questions.question_set import QuestionSet
from ..questions.submissions import Submission


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
    include_totals: bool = Field(
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
        with invalid_errors(data):
            return cls.model_validate(data)

    def write(self, questions: QuestionSet, marked: Iterable[Marked[Submission]]) -> str:
        """The marked submissions as CSV text, a row for each student in natural ID order. Raises
        ``GradeFlowError`` when two columns would share a name, such as a question named “Q1 marks”
        and Q1's marks."""
        header = self._header(questions)
        if repeated := next((column for column in header if header.count(column) > 1), None):
            raise GradeFlowError(f"The {quoted(repeated)} column would be written twice.")
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        for marked_submission in natsorted(marked, key=lambda marked: marked.sheet.student_id):
            writer.writerow(self._row(marked_submission, questions))
        return buffer.getvalue()

    def _header(self, questions: QuestionSet) -> list[str]:
        header = [self.student_id_column]
        for question_id in questions:
            header += [question_id] if self.include_answers else []
            header += [f"{question_id} marks"] if self.include_marks else []
            header += [f"{question_id} feedback"] if self.include_feedback else []
        header += ["total marks", "maximum marks"] if self.include_totals else []
        header += ["marking report"] if self.include_report else []
        return header

    def _row(self, marked: Marked[Submission], questions: QuestionSet) -> list[str]:
        cells = [marked.sheet.student_id]
        for question_id in questions:
            result = marked.results.get(question_id)
            cells += [marked.sheet.answers.get(question_id, "")] if self.include_answers else []
            if self.include_marks:
                cells.append(format_marks(result.marks) if result else "")
            cells += [result.feedback if result else ""] if self.include_feedback else []
        if self.include_totals:
            cells += [format_marks(marked.total_marks), format_marks(marked.max_marks)]
        if self.include_report:
            cells.append("\n".join(map(str, marked.report())))
        return cells
