import pytest

from gradeflow_core import QuestionSet
from gradeflow_core.editing import Position

QUESTIONS = QuestionSet.model_validate(
    {
        "Q1": {"type": "number"},
        "Q2": {"type": "values", "value_types": ["number", "text"]},
    }
)


@pytest.mark.parametrize(
    ("question", "value_number", "answer", "expected"),
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
    question: str | None, value_number: int | None, answer: str, expected: str
) -> None:
    assert Position(QUESTIONS, question, value_number).answer_value(answer) == expected
