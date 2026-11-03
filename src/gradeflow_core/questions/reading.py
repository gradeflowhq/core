"""How questions read answers: options people set, and the readers they configure."""

import math
import re
import textwrap
import unicodedata
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from natsort import natsorted, ns
from pydantic import BaseModel, ConfigDict, Field

from ..common.hints import ADVANCED_HINT
from ..common.text import as_text, quoted
from .types import Answer


class Reading(BaseModel):
    model_config = ConfigDict(title="Reading answers")

    blank_marker: str = Field(
        default="N/A",
        title="Blank answer marker",
        description="An answer written exactly like this counts as blank.",
        json_schema_extra=ADVANCED_HINT,
    )


class TextReading(Reading):
    trim_spaces: bool = Field(
        default=True,
        title="Ignore spaces at the ends",
        description="Spaces before and after an answer are removed before marking.",
        json_schema_extra=ADVANCED_HINT,
    )
    ignore_case: bool = Field(
        default=True,
        title="Ignore case",
        description="Answers are lowercased before marking.",
        json_schema_extra=ADVANCED_HINT,
    )
    ignore_punctuation: bool = Field(
        default=False,
        title="Ignore punctuation",
        description="Punctuation is removed before marking, so “(LL)” reads as “LL”.",
        json_schema_extra=ADVANCED_HINT,
    )

    def normalized(self, value: str) -> str:
        """``value``, such as a choice option, with its spaces, capitals and punctuation read as in
        answers."""
        value = value.strip() if self.trim_spaces else value
        value = value.lower() if self.ignore_case else value
        if self.ignore_punctuation:
            value = "".join(
                character
                for character in value
                if not unicodedata.category(character).startswith("P")
            )
        return value


class ListReading(TextReading):
    separator: str = Field(
        default=",",
        min_length=1,
        title="Separator",
        description="The character between values in an answer, such as a comma.",
        json_schema_extra=ADVANCED_HINT,
    )


def reading_field(factory: Callable[[], Reading]) -> Any:
    """A question's ``reading`` field, made by ``factory`` unless set."""
    return Field(
        default_factory=factory,
        title="Reading answers",
        description="How answers are read before marking.",
    )


def is_blank(answer: str, reading: Reading) -> bool:
    return answer in (reading.blank_marker, "")


def read_text(answer: str, reading: TextReading) -> str | None:
    # Case is changed after the blank check, so a blank marker in capitals still counts.
    if reading.trim_spaces:
        answer = answer.strip()
    return None if is_blank(answer, reading) else reading.normalized(answer)


def read_code(answer: str, reading: Reading) -> str | None:
    """The code as typed, without indentation every line shares."""
    code = textwrap.dedent("\n".join(line.rstrip() for line in answer.splitlines()).strip("\n"))
    return None if is_blank(code, reading) else code


def read_list(answer: str, reading: ListReading) -> list[str] | None:
    """The separated values of ``answer``, or None if it is blank."""
    if reading.trim_spaces:
        answer = answer.strip()
    if is_blank(answer, reading):
        return None
    values = answer.split(reading.separator)
    if reading.trim_spaces:
        values = [value.strip() for value in values]
    # Case is changed after each value's blank check, so a blank marker in capitals still counts.
    if reading.ignore_case:
        values = [value if is_blank(value, reading) else value.lower() for value in values]
    return values


def read_number(answer: str, reading: Reading) -> int | float | None:
    answer = answer.strip().translate(_MINUS_SIGN)
    if is_blank(answer, reading):
        return None
    try:
        return parse_number(answer)
    except ValueError:
        number_value = _number_with_unit(answer)
        if number_value is None:
            raise
        return number_value


def parse_number(text: str) -> int | float:
    """A whole number, decimal, scientific notation or plain fraction such as ``3/4``."""
    text = text.strip()
    for parse in (int, float, lambda written: float(Fraction(written))):
        try:
            value = parse(text)
        except ValueError, ZeroDivisionError:
            continue
        if math.isfinite(value):
            return value
        break
    raise ValueError(f"Couldn't read {quoted(text)} as a number.")


_MINUS_SIGN = str.maketrans("\u2212", "-")
"""The minus sign, as word processors and exam software write it, read as a hyphen."""
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPERSCRIPT_POWER = re.compile(r"10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_WRITTEN_OUT = re.compile(
    r"(?P<number>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s*[x×*]\s*10\s*(?:\^|\*\*)\s*(?P<power>[-+]?\d+))?"
    r"(?:\s?(?P<unit>(?:[^\W\d_]|[°%])[\w/^·*.°%-]{0,11}))?"
)


def _number_with_unit(text: str) -> float | None:
    """A number written with a unit, like `9.81 m/s^2`, or a power of ten, like `6.02 x 10^23`;
    none for anything else, such as `approx 9.8` or `9.81 or 10`."""
    text = _SUPERSCRIPT_POWER.sub(
        lambda power: "10^" + power[1].translate(_SUPERSCRIPT_DIGITS), text
    )
    match = _WRITTEN_OUT.fullmatch(text.translate(_SUPERSCRIPT_DIGITS))
    if match is None or not (match["power"] or match["unit"]):
        return None
    try:
        value = float(match["number"]) * 10 ** int(match["power"] or 0)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def write_answer(answer: Answer) -> str:
    """An answer as text: choices in order, values between commas, ``42`` not ``42.0``."""
    if answer is None:
        return ""
    if isinstance(answer, frozenset):
        return ", ".join(natsorted(answer, alg=ns.PRESORT))
    if isinstance(answer, list):
        return ", ".join(map(write_answer, answer))
    return as_text(answer)
