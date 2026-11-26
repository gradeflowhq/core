import pytest

from gradeflow_core import QuestionSet
from gradeflow_core.editing import Position
from gradeflow_core.steps.input_sources import AnswerPartSource

QUESTIONS = QuestionSet.load(
    {
        "Q1": {"type": "number"},
        "Q2": {
            "type": "multipart",
            "parts": [{"name": "amount", "type": "number"}, {"name": "unit"}],
        },
        "Q3": {
            "type": "list",
            "item": "parts",
            "parts": [{"name": "problem"}, {"name": "change"}],
        },
    }
)


@pytest.mark.parametrize(
    ("question_id", "part", "answer", "expected"),
    [
        ("Q1", None, " 42.0 ", ["42"]),
        ("Q1", None, "", []),
        ("Q1", None, "lots", []),
        ("Q2", "amount", "3.50, mol", ["3.5"]),
        ("Q2", "unit", "3.50, mol", ["mol"]),
        ("Q2", "unit", "3.50", []),
        (None, None, "anything", []),
    ],
)
def test_the_values_read_are_as_the_question_reads_them(
    question_id: str | None, part: str | None, answer: str, expected: list[str]
) -> None:
    position = Position(QUESTIONS, question_id)
    if part is not None:
        position, _ = position.reading(AnswerPartSource(part), {})
    assert position.values_read(answer) == expected


def test_a_check_of_items_reads_each_item_or_its_part() -> None:
    position = Position(QUESTIONS, "Q3").item_check_position
    assert position is not None
    answer = "overfitting, add L2, imbalance, , , "
    assert position.values_read(answer) == ["overfitting, add l2", "imbalance, "]
    change_position, refusal = position.reading(AnswerPartSource("change"), {})
    assert (refusal, change_position.input_kind) == (None, "part")
    assert change_position.values_read(answer) == ["add l2"]
    assert position.reading(AnswerPartSource("fix"), {})[1] == "The answer has no part named fix."
    assert position.input_kind == "item"


def test_only_a_list_question_has_an_item_check_position() -> None:
    assert Position(QUESTIONS, "Q1").item_check_position is None
    assert Position(QUESTIONS, None).item_check_position is None
