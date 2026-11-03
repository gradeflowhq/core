import pytest
from pydantic import ValidationError

from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.reading import ListReading


def test_choice_reads_options_ignoring_capitals_unless_set_otherwise() -> None:
    question = ChoiceQuestion(options=frozenset({"A", "b", "c"}))
    assert question.options == {"a", "b", "c"}
    assert question.read("A, B, C") == frozenset({"a", "b", "c"})


def test_choice_reads_with_a_custom_separator() -> None:
    question = ChoiceQuestion(
        options=frozenset({"a", "b", "c"}), reading=ListReading(separator="|")
    )
    options = question.read("Yes|NO|  Maybe ")
    assert options == frozenset({"yes", "no", "maybe"})


def test_choice_rejects_empty_option() -> None:
    with pytest.raises(ValidationError, match="Options can't be blank") as caught:
        ChoiceQuestion(options=frozenset({"A", ""}))
    assert caught.value.errors()[0]["loc"] == ("options",)


@pytest.mark.parametrize("options", [["A", "a"], ["a", " a "], ["a", "a"]])
def test_choice_rejects_options_read_alike(options: list[str]) -> None:
    with pytest.raises(ValidationError) as caught:
        ChoiceQuestion.model_validate({"options": options})
    [error] = caught.value.errors()
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
    option_schema = question.answer_schema()["properties"]["option"]
    assert option_schema["enum"] == ["a", "b", "c"]
    assert question.answer_values("B") == {"option": "b"}
    assert question.answer_values("z") == {"option": None}
    assert question.answer_from({"option": "b"}) == "b"
    assert question.answer_from({"option": None}) == ""


def test_a_choice_of_several_options_takes_them_in_order() -> None:
    question = ChoiceQuestion(options=frozenset({"a", "b", "c"}), allow_multiple=True)
    option_schema = question.answer_schema()["properties"]["options"]
    assert (
        option_schema["type"],
        option_schema["uniqueItems"],
        option_schema["items"]["enum"],
    ) == (
        "array",
        True,
        ["a", "b", "c"],
    )
    assert question.answer_values("C, a, z") == {"options": ["a", "c"]}
    assert question.answer_from({"options": ["c", "a"]}) == "a,c"
    assert question.answer_from({"options": []}) == ""


def test_choice_ignores_punctuation_when_set() -> None:
    question = ChoiceQuestion.model_validate(
        {"options": ["(a)", "b."], "reading": {"ignore_punctuation": True}}
    )
    assert question.options == {"a", "b"}
    assert question.read("(A), b") == frozenset({"a", "b"})
