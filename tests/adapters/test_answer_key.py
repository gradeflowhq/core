import pytest

from gradeflow_core import ChoiceQuestion, QuestionSet, Submission, read_answer_key


@pytest.mark.parametrize(
    ("cell", "answer"),
    [
        ("B", "B"),
        ("A,C,D", "A,C,D"),
        ("0.63|0.625|0.62|", "0.63"),
        (
            "Terminal|terminal|TERMINAL|~\u22121|-1|~Cutoff|cutoff|CUTOFF|~0",
            "Terminal~\u22121~Cutoff~0",
        ),
        ("2|2. from s0, transpose B and C", "2"),
        ("|1 |~x", "1 ~x"),
    ],
)
def test_each_blank_reads_as_its_first_accepted_answer(cell: str, answer: str) -> None:
    assert read_answer_key({"Q1": cell}).answers == {"Q1": answer}


def test_an_empty_cell_answers_nothing() -> None:
    assert read_answer_key({"Q1": "", "Q2": " "}).answers == {}


def test_the_answer_key_sets_questions_up_with_the_students_answers() -> None:
    students = [
        Submission(student_id=str(n), answers={"Q1": answer}) for n, answer in enumerate("AB")
    ]
    questions = QuestionSet({}).updated_from([*students, read_answer_key({"Q1": "A,C"})])
    question = questions["Q1"]
    assert isinstance(question, ChoiceQuestion)
    assert question.allow_multiple
