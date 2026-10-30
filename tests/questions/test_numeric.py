import pytest

from gradeflow_core.questions.number import NumberQuestion


@pytest.mark.parametrize(
    ("text", "number"), [("10", 10), ("3.14", 3.14), ("3/4", 0.75), ("1e3", 1000.0), ("-7", -7)]
)
def test_numeric_parses_numbers(text: str, number: float) -> None:
    assert NumberQuestion().read(text) == number


@pytest.mark.parametrize(
    ("text", "number"),
    [("\u22121", -1), ("\u22129.81 m/s^2", -9.81), ("1.6 x 10^\u221219", 1.6e-19)],
)
def test_the_minus_sign_reads_as_a_minus(text: str, number: float) -> None:
    assert NumberQuestion().read(text) == pytest.approx(number)


def test_numeric_raises_on_invalid() -> None:
    with pytest.raises(ValueError, match="Couldn't read"):
        NumberQuestion().read("not-a-number")


def test_numeric_blank_marker_returns_none() -> None:
    assert NumberQuestion().read("N/A") is None
    assert NumberQuestion().read("") is None


@pytest.mark.parametrize(
    ("written", "number"),
    [
        ("9.81 m/s^2", 9.81),
        ("-9.81 m/s²", -9.81),
        ("25.1cm", 25.1),
        ("37 °C", 37),
        ("6.022 x 10^23", 6.022e23),
        ("6.02×10²³", 6.02e23),
        ("1.6 * 10^-19 C", 1.6e-19),
    ],
)
def test_numbers_are_read_with_units_and_powers_of_ten_written_out(
    written: str, number: float
) -> None:
    assert NumberQuestion().read(written) == pytest.approx(number)


@pytest.mark.parametrize(
    "written", ["9,81", "approx 9.8", "9.81 or 10", "about ten", "3 apples and 2 pears"]
)
def test_answers_that_are_not_one_number_stay_unread(written: str) -> None:
    with pytest.raises(ValueError, match="Couldn't read"):
        NumberQuestion().read(written)
