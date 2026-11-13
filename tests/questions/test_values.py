import pytest

from gradeflow_core.questions.reading import ListReading
from gradeflow_core.questions.values import ValuesQuestion


def test_multiple_values_reads_text_and_numbers() -> None:
    question = ValuesQuestion(value_types=["text", "number", "text", "number", "text"])
    values = question.read("one, 2, three, 4.5, five")
    assert values == ["one", 2, "three", 4.5, "five"]


def test_values_with_a_custom_separator() -> None:
    question = ValuesQuestion(
        reading=ListReading(separator=";"),
        value_types=["text", "text", "number", "text"],
    )
    values = question.read("alpha; beta ; 3; gamma")
    assert values == ["alpha", "beta", 3, "gamma"]


def test_multiple_values_respects_value_types_and_blank_marker() -> None:
    question = ValuesQuestion(
        reading=ListReading(separator=";"),
        value_types=["number", "text", "number"],
    )
    values = question.read("1; N/A; ")
    assert values == [1, None, None]


def test_multiple_values_reads_fractions_and_scientific_notation() -> None:
    question = ValuesQuestion(value_types=["number", "number", "text"])
    values = question.read("3/2, 1e2, hello")
    assert values == [1.5, 100.0, "hello"]


def test_multiple_values_with_the_wrong_number_of_values_raises() -> None:
    question = ValuesQuestion(
        reading=ListReading(separator="|"),
        value_types=["text", "number", "text"],
    )
    with pytest.raises(ValueError, match="Expected 3 values"):
        question.read("only_one|2")


def test_multiple_values_trims_whitespace_and_uses_the_separator() -> None:
    question = ValuesQuestion(
        reading=ListReading(separator=";", trim_spaces=True),
        value_types=["text", "number", "text"],
    )
    values = question.read(" alpha ; 42 ;  beta  ")
    assert values == ["alpha", 42, "beta"]


def test_text_values_are_read_as_text() -> None:
    question = ValuesQuestion(
        value_types=["text", "number"],
        reading=ListReading(separator="~"),
    )
    assert question.read(" Yes ~ 2 ") == ["yes", 2]


def test_values_in_brackets_are_read_inside_them() -> None:
    coordinates = ValuesQuestion(value_types=["number", "number"])
    assert coordinates.read("(3, -2)") == coordinates.read("[3,-2]") == [3, -2]
    with pytest.raises(ValueError, match="Couldn't read"):
        coordinates.read("(3, -2")


def test_a_values_answer_is_written_value_by_value() -> None:
    question = ValuesQuestion(value_types=["number", "text"])
    titles = [
        value_schema["title"] for value_schema in question.answer_schema()["properties"].values()
    ]
    assert titles == ["Value 1, number", "Value 2, text"]
    assert question.answer_values("(3, x)") == {"value_1": "3", "value_2": "x"}
    assert question.answer_values(" ") == {"value_1": "", "value_2": ""}
    with pytest.raises(ValueError, match="Expected 2 values, but got 1"):
        question.answer_values("3")
    assert question.answer_from({"value_1": " 3", "value_2": "x "}) == "3,x"
    assert question.answer_from({"value_1": "", "value_2": " "}) == ""


def test_a_value_written_with_the_minus_sign_reads_as_a_number() -> None:
    question = ValuesQuestion(
        reading=ListReading(separator="~"), value_types=["text", "number", "number"]
    )
    assert question.read("terminal~\u22121~0") == ["terminal", -1, 0]
