"""How questions read answers: options people set, and the readers they configure."""

import math
import re
import textwrap
import unicodedata
from collections.abc import Callable, Iterable
from decimal import Decimal
from fractions import Fraction

from natsort import natsorted, ns
from pydantic import BaseModel, ConfigDict, Field

from ..common.hints import ADVANCED_HINT
from ..common.problems import InvalidError
from ..common.text import as_text, quoted
from .code_files import join_answer, split_answer
from .types import Answer


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", title="Reading answers")

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
        """``value``, such as a choice option, read as answers are: in Unicode NFC, then trimmed,
        lowercased and without punctuation as the reading sets."""
        value = unicodedata.normalize("NFC", value)
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


def reading_field[R: Reading](factory: Callable[[], R]) -> R:
    """A question's ``reading`` field, made by ``factory`` unless set."""
    return Field(
        default_factory=factory,
        title="Reading answers",
        description="How answers are read before marking.",
    )


def is_blank(answer: str, reading: Reading) -> bool:
    return answer in (reading.blank_marker, "")


def read_code(answer: str, reading: Reading) -> str | None:
    """The code as typed, without indentation every line of a file shares."""
    files = {
        path: textwrap.dedent("\n".join(line.rstrip() for line in source.splitlines()).strip("\n"))
        for path, source in split_answer(answer).items()
    }
    if all(is_blank(source, reading) for source in files.values()):
        return None
    try:
        return join_answer(files)
    except InvalidError as error:
        raise ValueError(str(error)) from None


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
        number = _number_with_unit(answer)
        if number is None:
            raise
        return number


def parse_number(text: str) -> int | float:
    """A whole number, decimal, scientific notation or plain fraction such as ``3/4``."""
    for parser in (int, float, lambda written: float(Fraction(written))):
        try:
            number = parser(text)
            if math.isfinite(number):
                return number
        except ValueError, ZeroDivisionError, OverflowError:
            continue
        break
    raise ValueError(f"Couldn't read {quoted(text)} as a number.")


_MINUS_SIGN = str.maketrans("\u2212", "-")
"""The minus sign, as word processors and exam software write it, read as a hyphen."""
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPERSCRIPT_POWER = re.compile(r"10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_NUMBER_WITH_UNIT = re.compile(
    r"(?P<number>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s*[x×*]\s*10\s*(?:\^|\*\*)\s*(?P<power>[-+]?\d+))?"
    r"(?:\s?(?P<unit>(?![x×]\d)(?:[^\W\d_]|[°%])[\w/^·*.°%-]{0,11}))?"
)


def _number_with_unit(text: str) -> float | None:
    """A number written with a unit, like `9.81 m/s^2`, or a power of ten, like `6.02 x 10^23`;
    none for anything else, such as `approx 9.8` or `9.81 or 10`."""
    text = _SUPERSCRIPT_POWER.sub(
        lambda power: "10^" + power[1].translate(_SUPERSCRIPT_DIGITS), text
    )
    match = _NUMBER_WITH_UNIT.fullmatch(text.translate(_SUPERSCRIPT_DIGITS))
    if match is None or not (match["power"] or match["unit"]):
        return None
    if match["unit"] and match["unit"][0].isnumeric():  # \w matches numerals such as ½
        return None
    try:
        number = float(Decimal(match["number"]).scaleb(int(match["power"] or 0)))
    except ArithmeticError, ValueError:
        return None
    return number if math.isfinite(number) else None


def sorted_options(options: Iterable[str]) -> list[str]:
    """``options`` in natural order, those that sort alike, such as ``1`` and ``01``, in one order
    whatever the hash seed."""
    return natsorted(options, alg=ns.PRESORT)


def write_answer(answer: Answer, reading: Reading | None = None) -> str:
    """An answer as text that ``reading`` reads back as the same answer: choices in order, values
    separated as it separates them, ``42`` not ``42.0``."""
    if answer is None:
        return ""
    separator = ", "
    if isinstance(reading, ListReading):
        separator = reading.separator + (" " if reading.trim_spaces else "")
    if isinstance(answer, frozenset):
        return separator.join(sorted_options(answer))
    if isinstance(answer, list):
        return separator.join(map(write_answer, answer))
    return as_text(answer)
