import math
import re

import pytest

from gradeflow_core.methods.answers.expressions import MAX_LENGTH, parse


def evaluate(text: str, **variables: float) -> float:
    return parse(text).value(variables)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("10 / 4", 2.5),
        ("(2 + 3) * 4", 20),
        ("2^3^2", 512),
        ("2**3", 8),
        ("-2^2", -4),
        ("2^-1", 0.5),
        ("+3 - -2", 5),
        ("1e-3", 0.001),
        ("2.5E2", 250),
        (".5", 0.5),
        ("3 × 4 ÷ 2 \N{MINUS SIGN} 1", 5),
        ("2·3", 6),
        ("3²", 9),
        ("2¹⁰", 1024),
        ("2⁻¹", 0.5),
        ("sqrt(16)", 4),
        ("ln(e)", 1),
        ("log(1000)", 3),
        ("exp(0) + abs(-2)", 3),
        ("sin(0) + cos(0) + tan(0)", 1),
        ("asin(1) + acos(1) + atan(0)", math.pi / 2),
        ("2pi", 2 * math.pi),
        ("π", math.pi),
        ("2e", 2 * math.e),
    ],
)
def test_reads_arithmetic_as_people_write_it(text: str, expected: float) -> None:
    assert evaluate(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2x", 6),
        ("2(x + 1)", 8),
        ("(x + 1)(x - 1)", 8),
        ("3 sin(x)", 3 * math.sin(3)),
        ("2x^2", 18),
        ("x(x)", 9),
        ("xy", 6),
        ("2xy²", 24),
        ("xe", 3 * math.e),
    ],
)
def test_letters_and_brackets_next_to_each_other_multiply(text: str, expected: float) -> None:
    assert evaluate(text, x=3, y=2) == pytest.approx(expected)


def test_values_are_floats() -> None:
    assert type(evaluate("2 * 3")) is float
    assert type(evaluate("abs(-2)")) is float


def test_variables_are_the_names_it_uses() -> None:
    assert parse("2x + 2").variables == {"x"}
    assert parse("sin(t) + e + pi + xy").variables == {"t", "x", "y"}
    assert parse("4").variables == set()


def test_names_as_written_keep_longer_names_and_brackets_name_anything() -> None:
    expression = parse("12 * side + Q1 + 2[1a] + [e] + e", names_as_written=True)
    assert expression.variables == {"side", "Q1", "1a", "e"}
    assert expression.value({"side": 2, "Q1": 3, "1a": 4, "e": 1}) == pytest.approx(36 + math.e)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("u2v", 2 * 3 * 4), ("x2", 2 * 3), ("u1v1 + 2", 3 * 4 + 2), ("x2^3", 3 * 8)],
)
def test_digits_in_a_name_are_numbers(text: str, expected: float) -> None:
    assert evaluate(text, u=3, v=4, x=3) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "expected"), [("x 2", 6), ("2 3", 6), ("x 2.5^2", 18.75), ("(x + 1) 2", 8)]
)
def test_a_number_after_a_name_bracket_or_number_multiplies(text: str, expected: float) -> None:
    assert evaluate(text, x=3) == pytest.approx(expected)


def test_a_name_splits_into_single_letters_as_sympy_splits_it() -> None:
    assert parse("pix").variables == {"p", "i", "x"}
    assert parse("2pi x").variables == {"x"}
    assert parse("sinx").variables == {"s", "i", "n", "x"}
    assert parse("2alpha + Beta + lambda").variables == {"alpha", "Beta", "l", "a", "m", "b", "d"}


def test_a_name_with_an_underscore_or_listed_is_one_variable() -> None:
    assert parse("u_1 v_1 + rate_2").variables == {"u_1", "v_1", "rate_2"}
    listed = frozenset({"u1", "v1", "side"})
    assert parse("u1 v1 + 2side x", variables=listed).variables == {"u1", "v1", "side", "x"}
    assert parse("u1v1", variables=listed).variables == {"u", "v"}
    assert parse("Q1 + Q1a", names_as_written=True).variables == {"Q1", "Q1a"}


def test_brackets_are_for_names_as_written_only() -> None:
    with pytest.raises(ValueError, match="Couldn't read"):
        parse("[x] + 1")


def test_says_which_text_it_couldnt_read() -> None:
    with pytest.raises(ValueError, match=re.escape("Couldn't read “2x +” as an expression.")):
        parse("2x +")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "()",
        "(x + 1",
        "x + 1)",
        "sin x",
        "sin()",
        "x.real",
        "x[0]",
        "a, b",
        "x = 1",
        "__import__('os')",
        "().__class__",
        "lambda: 1",
        "open('f')",
        "2 // 3",
        "½",
    ],
)
def test_refuses_anything_else(text: str) -> None:
    with pytest.raises(ValueError, match="Couldn't read"):
        parse(text)


def test_refuses_long_input_and_reads_deep_nesting_within_it() -> None:
    with pytest.raises(ValueError, match=f"longer than {MAX_LENGTH}"):
        parse("x+" * MAX_LENGTH + "x")
    depth = (MAX_LENGTH - 1) // 2
    assert evaluate("(" * depth + "1" + ")" * depth) == 1
    assert evaluate("-" * (MAX_LENGTH - 1) + "1") == -1


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1 / 0", "It divides by zero."),
        ("1 / (x - 3)", "It divides by zero."),
        ("sqrt(-1)", "can't take"),
        ("ln(0)", "can't take"),
        ("asin(2)", "can't take"),
        ("(-8)^(1/3)", "can't take"),
        ("9^9^9", "too large"),
        ("10^400", "too large"),
        ("1e308 * 10", "too large"),
        ("exp(1000)", "too large"),
        ("1e999 - 1e999", "too large"),
    ],
)
def test_values_it_cant_work_out_raise_value_errors(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        evaluate(text, x=3)
