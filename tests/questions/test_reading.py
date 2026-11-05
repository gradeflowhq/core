import pytest

from gradeflow_core.questions import reading
from gradeflow_core.questions.reading import ListReading


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("7", 7),
        ("2.5", 2.5),
        ("3/4", 0.75),
        ("1e5", 100000.0),
        ("-7", -7),
        ("-2.5", -2.5),
        ("  42  ", 42),
    ],
)
def test_number_reads_integers_decimals_and_fractions(text: str, number: float) -> None:
    assert reading.parse_number(text) == number


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "abc", "1/0"])
def test_number_rejects_what_is_not_a_finite_number(text: str) -> None:
    with pytest.raises(ValueError, match=f"Couldn't read “{text}” as a number"):
        reading.parse_number(text)


def test_read_list_follows_the_reading() -> None:
    answer = " a ,b,c "
    assert reading.read_list(answer, ListReading()) == ["a", "b", "c"]
    untrimmed = ListReading(trim_spaces=False)
    assert reading.read_list(answer, untrimmed) == [" a ", "b", "c "]
    assert reading.read_list("A|B", ListReading(separator="|")) == ["a", "b"]
    assert reading.read_list("A,B", ListReading(ignore_case=False)) == ["A", "B"]
    assert reading.read_list("only", ListReading()) == ["only"]


def test_na_and_empty_answers_are_blank() -> None:
    assert reading.is_blank("", reading.Reading()) is True
    assert reading.is_blank("N/A", reading.Reading()) is True
    assert reading.is_blank("hello", reading.Reading()) is False


def test_a_whole_number_too_large_for_a_float_is_not_a_number() -> None:
    with pytest.raises(ValueError, match="Couldn't read"):
        reading.parse_number("1" + "0" * 400)


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("6.02 x 10^23", 6.02e23),
        ("1.5 x 10^-3", 0.0015),
        ("6.02 × 10²³", 6.02e23),
        ("9.81 m/s^2", 9.81),
        ("30°", 30),
        ("5%", 5),
    ],
)
def test_a_number_written_out_reads_as_written(text: str, number: float) -> None:
    assert reading.read_number(text, reading.Reading()) == number


@pytest.mark.parametrize(
    "text",
    ["1 x 10^999999999", "1 x 10^400", "1" + "0" * 400 + " kg", "6.02x10-23", "2x10", "0x10", "3½"],
)
def test_what_only_starts_with_a_number_is_not_one(text: str) -> None:
    with pytest.raises(ValueError, match="Couldn't read"):
        reading.read_number(text, reading.Reading())
