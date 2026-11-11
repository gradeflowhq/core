import pytest

from gradeflow_core import QuestionSet
from gradeflow_core.editing import Position
from gradeflow_core.steps.input_sources import AnswerValueSource

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
    position = Position(QUESTIONS, question_id)
    if value_number is not None:
        position, _ = position.reading(AnswerValueSource(value_number), {})
    assert position.answer_value(answer) == expected
