import csv
from io import StringIO

import pytest

from gradeflow_core import (
    InvalidError,
    MarkedSubmission,
    Problem,
    QuestionResult,
    QuestionSet,
    ResultsCsv,
)
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

QUESTIONS = QuestionSet({"Q1": TextQuestion(), "Q2": NumberQuestion(max_marks=3)})
ROWS = [
    MarkedSubmission(
        student_id="10",
        answers={"Q1": "a", "Q2": "1"},
        results={
            "Q1": QuestionResult(proportion=1, max_marks=1, feedback="Correct."),
            "Q2": QuestionResult(proportion=1 / 3, max_marks=3),
        },
    ),
    MarkedSubmission(
        student_id="9",
        answers={"Q1": "b"},
        results={"Q1": QuestionResult(proportion=0, max_marks=1, feedback="Expected a.")},
    ),
]


def rows(options: ResultsCsv) -> list[list[str]]:
    return list(csv.reader(StringIO(options.write(QUESTIONS, ROWS))))


def test_a_row_for_each_student_in_natural_order_with_marks_worked_out() -> None:
    header, first, second = rows(ResultsCsv())
    assert header == [
        "student_id",
        "Q1",
        "Q1 marks",
        "Q1 feedback",
        "Q2",
        "Q2 marks",
        "Q2 feedback",
        "total marks",
        "maximum marks",
        "report",
    ]
    assert first == ["9", "b", "0", "Expected a.", "", "", "", "0", "4", "Q1 [0/1]\nExpected a."]
    assert second[:8] == ["10", "a", "1", "Correct.", "1", "1", "", "2"]


def test_columns_follow_the_options() -> None:
    header, *_ = rows(
        ResultsCsv(
            student_id_column="id",
            include_answers=False,
            include_feedback=False,
            include_total=False,
            include_report=False,
        )
    )
    assert header == ["id", "Q1 marks", "Q2 marks"]


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_the_student_id_column_needs_a_name(name: str) -> None:
    with pytest.raises(InvalidError) as caught:
        ResultsCsv.load({"student_id_column": name})
    assert caught.value.problems == [Problem("Required.", field=("student_id_column",))]
