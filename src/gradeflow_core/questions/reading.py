"""How questions read answers: options people set, and the readers they configure."""

import math
import re
import textwrap
import unicodedata
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from natsort import natsorted
from pydantic import BaseModel, ConfigDict, Field

from ..common.hints import ADVANCED, hint
from ..common.text import as_text, quoted
from .layouts import guess_layouts
from .types import Answer


class Reading(BaseModel):
    model_config = ConfigDict(title="Reading answers")

    blank_marker: str = Field(
        default="N/A",
        title="Blank answer marker",
        description="An answer written exactly like this counts as blank.",
        json_schema_extra=ADVANCED,
    )


class TextReading(Reading):
    trim_spaces: bool = Field(
        default=True,
        title="Ignore spaces at the ends",
        description="Spaces before and after an answer are removed before marking.",
        json_schema_extra=ADVANCED,
    )
    ignore_case: bool = Field(
        default=True,
        title="Ignore capitals",
        description="Answers are changed to small letters before marking.",
        json_schema_extra=ADVANCED,
    )
    ignore_punctuation: bool = Field(
        default=False,
        title="Ignore punctuation",
        description="Punctuation is removed before marking, so “(LL)” reads as “LL”.",
        json_schema_extra=ADVANCED,
    )

    def option(self, value: str) -> str:
        """``value``, such as a choice option, with its spaces and capitals read as in answers."""
        value = value.strip() if self.trim_spaces else value
        return value.lower() if self.ignore_case else value


class ListReading(TextReading):
    separator: str = Field(
        default=",",
        min_length=1,
        title="Separator",
        description="The character between values in an answer, such as a comma.",
        json_schema_extra=ADVANCED,
    )


class CodeReading(Reading):
    guess_indentation: bool = Field(
        default=False,
        title="Guess lost indentation",
        description="Code whose indentation was lost reads as the layouts it most likely had.",
        json_schema_extra=ADVANCED,
    )
    most_guesses: int = Field(
        default=3,
        ge=1,
        le=10,
        title="Most guesses",
        description="How many layouts an answer reads as, at most.",
        json_schema_extra=hint(advanced=True, applies_when={"guess_indentation": True}),
    )


def reading_field(factory: Callable[[], Reading]) -> Any:
    """A question's ``reading`` field, made by ``factory`` unless set."""
    return Field(
        default_factory=factory,
        title="Reading answers",
        description="How answers are read before marking.",
    )


def is_blank(raw: str, reading: Reading) -> bool:
    return raw in (reading.blank_marker, "")


def read_text(raw: str, reading: TextReading) -> str | None:
    # Case is changed after the blank check, so a blank marker in capitals still counts.
    if reading.trim_spaces:
        raw = raw.strip()
    if is_blank(raw, reading):
        return None
    if reading.ignore_case:
        raw = raw.lower()
    if reading.ignore_punctuation:
        raw = "".join(c for c in raw if not unicodedata.category(c).startswith("P"))
    return raw


def read_code(raw: str, reading: CodeReading) -> str | list[str] | None:
    """The code as typed, without indentation every line shares; with guessing, its likeliest
    layouts, or the code alone when none compiles."""
    code = textwrap.dedent("\n".join(line.rstrip() for line in raw.splitlines()).strip("\n"))
    if is_blank(code, reading):
        return None
    if not reading.guess_indentation:
        return code
    return guess_layouts(code, reading.most_guesses) or [code]


def read_list(raw: str, reading: ListReading) -> list[str] | None:
    """The separated values of ``raw``, or None if it is blank."""
    if reading.trim_spaces:
        raw = raw.strip()
    if is_blank(raw, reading):
        return None
    values = raw.split(reading.separator)
    if reading.trim_spaces:
        values = [value.strip() for value in values]
    # Case is changed after each value's blank check, so a blank marker in capitals still counts.
    if reading.ignore_case:
        values = [value if is_blank(value, reading) else value.lower() for value in values]
    return values


def read_number(raw: str, reading: Reading) -> int | float | None:
    raw = raw.strip().translate(_MINUS)
    if is_blank(raw, reading):
        return None
    try:
        return number(raw)
    except ValueError:
        written = _written_out(raw)
        if written is None:
            raise
        return written


def number(text: str) -> int | float:
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


_MINUS = str.maketrans("\u2212", "-")
"""The minus sign, as word processors and exam software write it, read as a hyphen."""
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPERSCRIPT_POWER = re.compile(r"10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_WRITTEN_OUT = re.compile(
    r"(?P<number>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s*[x×*]\s*10\s*(?:\^|\*\*)\s*(?P<power>[-+]?\d+))?"
    r"(?:\s?(?P<unit>(?:[^\W\d_]|[°%])[\w/^·*.°%-]{0,11}))?"
)


def _written_out(text: str) -> float | None:
    """A number written with a unit, like `9.81 m/s^2`, or a power of ten, like `6.02 x 10^23`;
    none for anything else, such as `approx 9.8` or `9.81 or 10`."""
    text = _SUPERSCRIPT_POWER.sub(lambda power: "10^" + power[1].translate(_SUPERSCRIPT), text)
    found = _WRITTEN_OUT.fullmatch(text.translate(_SUPERSCRIPT))
    if found is None or not (found["power"] or found["unit"]):
        return None
    try:
        value = float(found["number"]) * 10 ** int(found["power"] or 0)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def write_answer(answer: Answer) -> str:
    """An answer as text: choices in order, values between commas, ``42`` not ``42.0``."""
    if answer is None:
        return ""
    if isinstance(answer, frozenset):
        return ", ".join(natsorted(answer))
    if isinstance(answer, list):
        return ", ".join(map(write_answer, answer))
    return as_text(answer)
