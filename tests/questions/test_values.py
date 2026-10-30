import pytest

from gradeflow_core.questions.reading import ListReading
from gradeflow_core.questions.values import ValuesQuestion


def test_multi_values_parses_and_filters_numbers() -> None:
    q = ValuesQuestion(value_types=["text", "number", "text", "number", "text"])
    parsed = q.read("one, 2, three, 4.5, five")
    assert parsed == ["one", 2, "three", 4.5, "five"]


def test_multi_values_with_custom_options() -> None:
    q = ValuesQuestion(
        reading=ListReading(separator=";"),
        value_types=["text", "text", "number", "text"],
    )
    parsed = q.read("alpha; beta ; 3; gamma")
    assert parsed == ["alpha", "beta", 3, "gamma"]


def test_multi_values_respects_value_types_and_blank_marker() -> None:
    q = ValuesQuestion(
        reading=ListReading(separator=";"),
        value_types=["number", "text", "number"],
    )
    parsed = q.read("1; N/A; ")
    assert parsed == [1, None, None]


def test_multi_values_parses_fraction_and_scientific_notation() -> None:
    q = ValuesQuestion(value_types=["number", "number", "text"])
    parsed = q.read("3/2, 1e2, hello")
    assert parsed == [1.5, 100.0, "hello"]


def test_multi_values_cardinality_mismatch_raises() -> None:
    q = ValuesQuestion(
        reading=ListReading(separator="|"),
        value_types=["text", "number", "text"],
    )
    with pytest.raises(ValueError, match="Expected 3 values"):
        q.read("only_one|2")


def test_multi_values_trims_whitespace_and_uses_delimiter() -> None:
    q = ValuesQuestion(
        reading=ListReading(separator=";", trim_spaces=True),
        value_types=["text", "number", "text"],
    )
    parsed = q.read(" alpha ; 42 ;  beta  ")
    assert parsed == ["alpha", 42, "beta"]


def test_multi_values_apply_text_config_to_text_slots() -> None:
    q = ValuesQuestion(
        value_types=["text", "number"],
        reading=ListReading(separator="~"),
    )
    assert q.read(" Yes ~ 2 ") == ["yes", 2]


def test_values_in_brackets_are_read_inside_them() -> None:
    point = ValuesQuestion(value_types=["number", "number"])
    assert point.read("(3, -2)") == point.read("[3,-2]") == [3, -2]
    with pytest.raises(ValueError, match="Couldn't read"):
        point.read("(3, -2")


def test_a_values_answer_is_written_value_by_value() -> None:
    question = ValuesQuestion(value_types=["number", "text"])
    titles = [part["title"] for part in question.answer_schema()["properties"].values()]
    assert titles == ["Value 1, number", "Value 2, text"]
    assert question.answer_values("(3, x)") == {"value_1": "3", "value_2": "x"}
    assert question.answer_values("3") == {"value_1": "3", "value_2": ""}
    assert question.answer_from({"value_1": " 3", "value_2": "x "}) == "3,x"
    assert question.answer_from({"value_1": "", "value_2": " "}) == ""


def test_a_value_written_with_the_minus_sign_reads_as_a_number() -> None:
    q = ValuesQuestion(reading=ListReading(separator="~"), value_types=["text", "number", "number"])
    assert q.read("terminal~\u22121~0") == ["terminal", -1, 0]
