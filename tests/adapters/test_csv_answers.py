import re

import pytest

from gradeflow_core import AnswersCsv, GradeFlowError, InvalidError, Problem


def test_csv_adapter_reads_answers_by_student() -> None:
    csv_text = "student_id,q1,q2\ns1,foo,bar\ns2,baz,qux\n"
    subs = AnswersCsv(student_id_column="student_id", answer_columns=["q1", "q2"]).read(csv_text)
    assert len(subs) == 2
    assert subs[0].student_id == "s1"
    assert subs[0].answers == {"q1": "foo", "q2": "bar"}
    assert subs[1].answers == {"q1": "baz", "q2": "qux"}


def test_csv_adapter_reads_only_the_chosen_columns() -> None:
    csv_text = "id,a,b,extra\n123,x,y,z\n456,m,n,o\n"
    subs = AnswersCsv(student_id_column="id", answer_columns=["a", "b"]).read(csv_text)
    assert len(subs) == 2
    assert subs[0].student_id == "123"
    assert subs[0].answers == {"a": "x", "b": "y"}
    assert subs[1].answers == {"a": "m", "b": "n"}


def test_csv_adapter_missing_student_id_names_the_row() -> None:
    csv_text = "student_id,ans\ns1,ok\n,empty\n"
    with pytest.raises(GradeFlowError) as exc_info:
        AnswersCsv(student_id_column="student_id", answer_columns=["ans"]).read(csv_text)
    assert str(exc_info.value) == "Row 3 has no student ID in the “student_id” column."


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("student_id,q1,pts\n", "The file has no “q2” column."),
        ("id,q1\n", "The file has no “student_id”, “q2” and “pts” columns."),
    ],
)
def test_csv_adapter_names_missing_columns(header: str, message: str) -> None:
    config = AnswersCsv(
        student_id_column="student_id", answer_columns=["q1", "q2"], marks_columns={"q1": "pts"}
    )
    with pytest.raises(GradeFlowError) as exc_info:
        config.read(header + "s1,a\n")
    assert str(exc_info.value) == message


def test_csv_adapter_reads_an_empty_file_as_no_students() -> None:
    config = AnswersCsv(student_id_column="id", answer_columns=["q1"])
    assert config.read("") == []


def test_csv_adapter_rejects_duplicate_student_ids_naming_their_rows() -> None:
    csv_text = "id,q1\na,1\nb,2\na,3\nc,4\nb,5\na,6\n"
    with pytest.raises(GradeFlowError) as exc_info:
        AnswersCsv(student_id_column="id", answer_columns=["q1"]).read(csv_text)
    assert str(exc_info.value) == (
        "Some students are in more than one row: rows 2, 4 and 7; rows 3 and 6."
    )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            "s2,answer",
            "Row 3 has no “q2” cell. A line break in a cell without quotes can cause this.",
        ),
        ("s2", "Row 3 has no “q1” and “q2” cells."),
        (",a,b", "Row 3 has no student ID"),
    ],
)
def test_csv_adapter_numbers_rows_not_lines(row: str, message: str) -> None:
    csv_text = f'student_id,q1,q2\ns1,"two\nlines",b\n{row}\n'
    with pytest.raises(GradeFlowError, match=f"^{re.escape(message)}"):
        AnswersCsv(student_id_column="student_id", answer_columns=["q1", "q2"]).read(csv_text)


def test_csv_adapter_preserves_empty_strings() -> None:
    csv_text = "student_id,a,b\ns1,,val\ns2,val2,\n"
    subs = AnswersCsv(student_id_column="student_id", answer_columns=["a", "b"]).read(csv_text)
    assert len(subs) == 2
    assert subs[0].answers == {"a": "", "b": "val"}
    assert subs[1].answers == {"a": "val2", "b": ""}


def test_csv_adapter_reads_marks_columns() -> None:
    csv_text = "student_id,q1,q2,score_q1\ns1,yes,no,3.5\ns2,no,yes,0.0\n"
    subs = AnswersCsv(
        student_id_column="student_id",
        answer_columns=["q1", "q2"],
        marks_columns={"q1": "score_q1"},
    ).read(csv_text)
    assert subs[0].answers == {"q1": "yes", "q2": "no"}
    assert subs[0].marks == {"q1": 3.5}
    assert subs[1].marks == {"q1": 0.0}


@pytest.mark.parametrize("cell", ["N/A", "-", "1,5", "notanumber", "", "  ", "nan", "inf"])
def test_csv_adapter_marks_that_are_not_numbers_are_no_marks(cell: str) -> None:
    subs = AnswersCsv(
        student_id_column="student_id", answer_columns=["q1"], marks_columns={"q1": "pts"}
    ).read(f'student_id,q1,pts\ns1,ans,"{cell}"\n')
    assert subs[0].marks == {}


def test_csv_adapter_marks_may_have_spaces() -> None:
    subs = AnswersCsv(
        student_id_column="student_id", answer_columns=["q1"], marks_columns={"q1": "pts"}
    ).read("student_id,q1,pts\ns1,ans, 1.5 \n")
    assert subs[0].marks == {"q1": 1.5}


def test_options_load_from_data_and_say_what_is_missing() -> None:
    data = {"student_id_column": "id", "answer_columns": ["Q1"]}
    assert AnswersCsv.load(data) == AnswersCsv(student_id_column="id", answer_columns=["Q1"])
    with pytest.raises(InvalidError) as caught:
        AnswersCsv.load({"student_id_column": "id"})
    assert caught.value.problems == [
        Problem("Required.", field=("answer_columns",), incomplete=True)
    ]
