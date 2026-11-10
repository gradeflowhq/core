import pytest

from gradeflow_core import QuestionSet
from gradeflow_core.editing import Position

QUESTIONS = QuestionSet.load(
    {
        "Q1": {"type": "number"},
        "Q2": {"type": "values", "value_types": ["number", "text"]},
    }
)


@pytest.mark.parametrize(
    ("question_id", "value_number", "answer", "expected"),
    [
        ("Q1", None, " 42.0 ", "42"),
        ("Q1", None, "", ""),
        ("Q1", None, "lots", ""),
        ("Q2", 1, "3.50, mol", "3.5"),
        ("Q2", 2, "3.50, mol", "mol"),
        ("Q2", 2, "3.50", ""),
        (None, None, "anything", ""),
    ],
)
def test_the_answer_value_is_the_value_as_the_question_reads_it(
    question_id: str | None, value_number: int | None, answer: str, expected: str
) -> None:
    assert Position(QUESTIONS, question_id, value_number).answer_value(answer) == expected
