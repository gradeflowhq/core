import csv
import math
from io import StringIO

from pydantic import BaseModel, ConfigDict, Field

from ..common.exceptions import GradeFlowError
from ..common.text import plain_join, plural, quoted_join
from ..questions.submissions import Submission
from ..questions.types import QuestionId


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

    def read(self, text: str) -> list[Submission]:
        """Each row's submission, as the CSV ``text`` holds it: its answers, and any marks already
        given. Raises ``GradeFlowError`` for a file it can't read."""
        reader = csv.DictReader(StringIO(text.removeprefix("\ufeff"), newline=""))
        try:
            header, rows = reader.fieldnames, list(reader)
        except csv.Error:
            # Read with newline="", the only error csv raises is for a cell over its size limit.
            raise GradeFlowError(
                "A cell is too long to read. A quote mark left open can cause this."
            ) from None
        if header is None:
            return []
        columns = dict.fromkeys(
            [self.student_id_column, *self.answer_columns, *self.marks_columns.values()]
        )
        missing_columns = [column for column in columns if column not in header]
        if missing_columns:
            noun = plural("column", len(missing_columns))
            raise GradeFlowError(
                f"The file has no {quoted_join(missing_columns, conjunction='and')} {noun}."
            )
        repeated_columns = [column for column in columns if header.count(column) > 1]
        if repeated_columns:
            noun, verb = ("column", "is") if len(repeated_columns) == 1 else ("columns", "are")
            repeated_names = quoted_join(repeated_columns, conjunction="and")
            raise GradeFlowError(f"The {repeated_names} {noun} {verb} written twice.")

        submissions: list[Submission] = []
        row_numbers_by_student_id: dict[str, list[int]] = {}
        for row_number, row in enumerate(rows, start=2):
            # DictReader fills missing cells with None, typically after an unquoted newline.
            missing_cells = [column for column, value in row.items() if value is None]
            if missing_cells:
                noun = plural("cell", len(missing_cells))
                missing_names = quoted_join(missing_cells, conjunction="and")
                raise GradeFlowError(
                    f"Row {row_number} has no {missing_names} {noun}. "
                    "A line break in a cell without quotes can cause this."
                )
            student_id = row[self.student_id_column]
            if not student_id:
                raise GradeFlowError(
                    f"Row {row_number} has no student ID in the “{self.student_id_column}” column."
                )
            row_numbers_by_student_id.setdefault(student_id, []).append(row_number)
            submissions.append(
                Submission(
                    student_id=student_id,
                    answers={column: row[column] for column in self.answer_columns},
                    marks=_given_marks(row, self.marks_columns),
                )
            )
        duplicate_row_numbers = [
            row_numbers
            for row_numbers in row_numbers_by_student_id.values()
            if len(row_numbers) > 1
        ]
        if duplicate_row_numbers:
            duplicate_rows = "; ".join(
                f"rows {plain_join(row_numbers, conjunction='and')}"
                for row_numbers in duplicate_row_numbers
            )
            raise GradeFlowError(f"Some students are in more than one row: {duplicate_rows}.")
        return submissions


def _given_marks(
    row: dict[str, str], marks_columns: dict[QuestionId, str]
) -> dict[QuestionId, float]:
    """The marks in ``marks_columns``; a cell that isn't a number, such as "-", has none."""
    marks_by_question: dict[QuestionId, float] = {}
    for question_id, column in marks_columns.items():
        try:
            marks = float(row[column])
        except ValueError:
            continue
        if math.isfinite(marks):
            marks_by_question[question_id] = marks
    return marks_by_question
