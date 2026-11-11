import csv
from io import StringIO

import pytest

from gradeflow_core import (
    GradeFlowError,
    InvalidError,
    Marked,
    Problem,
    QuestionResult,
    QuestionSet,
    ResultsCsv,
    Submission,
)
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

QUESTIONS = QuestionSet({"Q1": TextQuestion(), "Q2": NumberQuestion(max_marks=3)})
MARKED_SUBMISSIONS = [
    Marked(
        sheet=Submission(student_id="10", answers={"Q1": "a", "Q2": "1"}),
        results={
            "Q1": QuestionResult(proportion=1, max_marks=1, feedback="Correct."),
            "Q2": QuestionResult(proportion=1 / 3, max_marks=3),
        },
        max_marks=4,
    ),
    Marked(
        sheet=Submission(student_id="9", answers={"Q1": "b"}),
        results={"Q1": QuestionResult(proportion=0, max_marks=1, feedback="Expected a.")},
        max_marks=4,
    ),
]


def written_rows(options: ResultsCsv) -> list[list[str]]:
    return list(csv.reader(StringIO(options.write(QUESTIONS, MARKED_SUBMISSIONS))))


def test_a_row_for_each_student_in_natural_order_with_marks_worked_out() -> None:
    header, first, second = written_rows(ResultsCsv())
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
        "marking report",
    ]
    assert first == ["9", "b", "0", "Expected a.", "", "", "", "0", "4", "Q1 [0/1]\nExpected a."]
    assert second[:8] == ["10", "a", "1", "Correct.", "1", "1", "", "2"]


def test_columns_follow_the_options() -> None:
    header, *_ = written_rows(
        ResultsCsv(
            student_id_column="id",
            include_answers=False,
            include_feedback=False,
            include_totals=False,
            include_report=False,
        )
    )
    assert header == ["id", "Q1 marks", "Q2 marks"]


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_the_student_id_column_needs_a_name(name: str) -> None:
    with pytest.raises(InvalidError) as exc_info:
        ResultsCsv.load({"student_id_column": name})
    assert exc_info.value.problems == [Problem("Required.", field=("student_id_column",))]


@pytest.mark.parametrize(
    ("questions", "column"),
    [
        (QuestionSet({"Q1": TextQuestion(), "Q1 marks": TextQuestion()}), "Q1 marks"),
        (QuestionSet({"student_id": TextQuestion()}), "student_id"),
    ],
)
def test_a_column_name_used_twice_is_refused(questions: QuestionSet, column: str) -> None:
    with pytest.raises(GradeFlowError) as exc_info:
        ResultsCsv().write(questions, [])
    assert str(exc_info.value) == f"The “{column}” column would be written twice."
