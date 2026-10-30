import pytest
from pydantic import ValidationError

from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.reading import ListReading


def test_choice_reads_options_ignoring_capitals_unless_set_otherwise() -> None:
    question = ChoiceQuestion(options=frozenset({"A", "b", "c"}))
    assert question.options == {"a", "b", "c"}
    assert question.read("A, B, C") == frozenset({"a", "b", "c"})


def test_choice_normalize_and_custom_delimiter() -> None:
    q = ChoiceQuestion(options=frozenset({"a", "b", "c"}), reading=ListReading(separator="|"))
    parsed = q.read("Yes|NO|  Maybe ")
    assert parsed == frozenset({"yes", "no", "maybe"})


def test_choice_rejects_empty_option() -> None:
    with pytest.raises(ValidationError, match="Options can't be blank") as exc_info:
        ChoiceQuestion(options=frozenset({"A", ""}))
    assert exc_info.value.errors()[0]["loc"] == ("options",)


@pytest.mark.parametrize("options", [["A", "a"], ["a", " a "], ["a", "a"]])
def test_choice_rejects_options_read_alike(options: list[str]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ChoiceQuestion.model_validate({"options": options})
    [error] = exc_info.value.errors()
    assert error["loc"] == ("options",)
    assert error["msg"] == "Value error, Options must be different: “a” appears twice."


def test_choice_options_are_read_like_answers() -> None:
    question = ChoiceQuestion(options=frozenset({" A", "b "}))
    assert question.options == {"a", "b"}


@pytest.mark.parametrize("kwargs", [{}, {"options": []}, {"options": set()}])
def test_choice_requires_at_least_one_option(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion.model_validate(kwargs)


def test_a_choice_answer_is_written_by_choosing_options() -> None:
    question = ChoiceQuestion(options=frozenset({"b", "a", "c"}))
    part = question.answer_schema()["properties"]["option"]
    assert part["enum"] == ["a", "b", "c"]
    assert question.answer_values("B") == {"option": "b"}
    assert question.answer_values("z") == {"option": None}
    assert question.answer_from({"option": "b"}) == "b"
    assert question.answer_from({"option": None}) == ""


def test_a_choice_of_several_options_takes_them_in_order() -> None:
    question = ChoiceQuestion(options=frozenset({"a", "b", "c"}), allow_multiple=True)
    part = question.answer_schema()["properties"]["options"]
    assert (part["type"], part["uniqueItems"], part["items"]["enum"]) == (
        "array",
        True,
        ["a", "b", "c"],
    )
    assert question.answer_values("C, a, z") == {"options": ["a", "c"]}
    assert question.answer_from({"options": ["c", "a"]}) == "a,c"
    assert question.answer_from({"options": []}) == ""
