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
    ("question", "value", "answer", "part"),
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
def test_the_answer_value_is_what_a_check_there_reads_written_alike(
    question: str | None, value: int | None, answer: str, part: str
) -> None:
    assert Position(QUESTIONS, question, value).answer_value(answer) == part
