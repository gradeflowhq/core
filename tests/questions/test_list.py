import pytest

from gradeflow_core.common.problems import InvalidError
from gradeflow_core.questions.list import ListQuestion
from gradeflow_core.questions.multipart import MultipartQuestion
from gradeflow_core.questions.number import NumberQuestion

PROBLEMS_AND_CHANGES = {
    "item": "parts",
    "parts": [{"name": "problem"}, {"name": "change"}],
    "reading": {"separator": "~"},
}


def test_a_list_reads_its_items_without_blank_ones() -> None:
    assert ListQuestion().read("prophase, , metaphase, N/A") == ["prophase", "metaphase"]
    assert ListQuestion(item="number").read("(3, -2)") == [3, -2]
    assert ListQuestion().read(" , ") is None


def test_items_in_parts_keep_blank_parts_and_drop_blank_items() -> None:
    question = ListQuestion.load(PROBLEMS_AND_CHANGES)
    assert question.read("Overfitting~add L2~~~Imbalance~") == [
        {"problem": "overfitting", "change": "add l2"},
        {"problem": "imbalance", "change": None},
    ]
    with pytest.raises(ValueError, match="Expected 2 parts for each item, but got 3 values"):
        question.read("a~b~c")


def test_items_in_parts_need_two_or_more_parts() -> None:
    with pytest.raises(InvalidError, match="/parts: Add at least 2"):
        ListQuestion.load({"item": "parts"})
    assert ListQuestion.load({"item": "text", "parts": [{"name": "unused"}]}).item == "text"


def test_an_item_is_read_as_one_value_or_as_a_multipart_question() -> None:
    assert isinstance(ListQuestion(item="number").item_question(), NumberQuestion)
    item_question = ListQuestion.load(PROBLEMS_AND_CHANGES).item_question()
    assert isinstance(item_question, MultipartQuestion)
    assert [part.name for part in item_question.parts] == ["problem", "change"]


def test_a_list_is_written_item_by_item() -> None:
    question = ListQuestion.load(PROBLEMS_AND_CHANGES)
    assert question.answer_schema()["properties"]["items"]["items"]["properties"] == {
        "problem": {"type": "string", "title": "Problem"},
        "change": {"type": "string", "title": "Change"},
    }
    values = question.answer_values("Overfitting~add L2~~")
    assert values == {"items": [{"problem": "Overfitting", "change": "add L2"}]}
    assert question.answer_from({"items": [*values["items"], {"problem": " ", "change": ""}]}) == (
        "Overfitting~add L2"
    )
    assert ListQuestion().answer_values("a, , b") == {"items": ["a", "b"]}
    assert ListQuestion().answer_from({"items": ["a", " ", "b "]}) == "a,b"


def test_a_list_is_written_back_with_its_separator() -> None:
    assert ListQuestion.load(PROBLEMS_AND_CHANGES).normalize("A~b~C~d") == "a~ b~ c~ d"
