import csv
import math
from io import StringIO

from pydantic import BaseModel, ConfigDict, Field

from ..common.exceptions import GradeFlowError
from ..common.problems import as_invalid
from ..common.text import plain_join, plural, quoted_join
from ..questions.submissions import StudentId, Submission
from ..questions.types import QuestionId
from .errors import adapter_errors


class AnswersCsv(BaseModel):
    """Where a CSV of answers holds each student's ID, answers and any marks already given."""

    model_config = ConfigDict(title="CSV answers")

    student_id_column: str = Field(
        title="Student ID column", description="The column that holds each student's ID."
    )
    answer_columns: list[str] = Field(
        title="Answer columns", description="The columns that hold answers, one per question."
    )
    marks_columns: dict[QuestionId, str] = Field(
        default_factory=dict,
        title="Marks columns",
        description="For each question, the column with marks already given.",
    )

    @classmethod
    def load(cls, data: object) -> AnswersCsv:
        """``data`` as options. Raises ``InvalidError`` if they aren't valid."""
        with as_invalid(data):
            return cls.model_validate(data)

    @adapter_errors()
    def read(self, text: str) -> list[Submission]:
        """Each row's submission, as the CSV ``text`` holds it: its answers, and any marks already
        given. Raises ``GradeFlowError`` for a file it can't read."""
        reader = csv.DictReader(StringIO(text))
        if reader.fieldnames is None:
            return []
        columns = dict.fromkeys(
            [self.student_id_column, *self.answer_columns, *self.marks_columns.values()]
        )
        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            noun = plural("column", len(missing))
            raise GradeFlowError(
                f"The file has no {quoted_join(missing, conjunction='and')} {noun}."
            )

        students: list[Submission] = []
        rows_by_id: dict[StudentId, list[int]] = {}
        for row_number, row in enumerate(reader, start=2):
            # DictReader fills missing cells with None, typically after an unquoted newline.
            missing_cells = [column for column, value in row.items() if value is None]
            if missing_cells:
                noun = plural("cell", len(missing_cells))
                named = quoted_join(missing_cells, conjunction="and")
                raise GradeFlowError(
                    f"Row {row_number} has no {named} {noun}. "
                    "A line break in a cell without quotes can cause this."
                )
            student_id = row[self.student_id_column]
            if not student_id:
                raise GradeFlowError(
                    f"Row {row_number} has no student ID in the “{self.student_id_column}” column."
                )
            rows_by_id.setdefault(student_id, []).append(row_number)
            students.append(
                Submission(
                    student_id=student_id,
                    answers={column: row[column] for column in self.answer_columns},
                    marks=_marks(row, self.marks_columns),
                )
            )
        duplicates = [rows for rows in rows_by_id.values() if len(rows) > 1]
        if duplicates:
            groups = "; ".join(f"rows {plain_join(rows, conjunction='and')}" for rows in duplicates)
            raise GradeFlowError(f"Some students are in more than one row: {groups}.")
        return students


def _marks(row: dict[str, str], columns: dict[QuestionId, str]) -> dict[QuestionId, float]:
    """The marks in ``columns``; a cell that isn't a number, such as "-", has none."""
    marks: dict[QuestionId, float] = {}
    for key, column in columns.items():
        try:
            value = float(row[column])
        except ValueError:
            continue
        if math.isfinite(value):
            marks[key] = value
    return marks
