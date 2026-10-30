import re
import textwrap
from collections.abc import Iterable
from decimal import Decimal
from typing import Literal

from .latex import latex_text

_BACKTICK_RUN = re.compile(r"`+")
_CODE_SPAN = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)", re.DOTALL)
# An escaped character, or a formula as Markdown's dollar math reads one: in $$ on lines of its
# own, or in $ with no space just inside and no digit just outside, so "$5 or $6" stays text.
_ESCAPE_OR_FORMULA = re.compile(
    r"\\([!-/:-@\[-`{-~])"
    r"|^\$\$((?:\\.|[^\\$])+)\$\$$"
    r"|(?<!\d)\$(?!\s)((?:\\.|[^\\$])+)(?<!\s)\$(?!\d)",
    re.MULTILINE,
)


_FRACTIONS = {0.5: "½", 0.25: "¼", 0.75: "¾"}


def format_number(value: float) -> str:
    """``value`` as people write it: 3 not 3.0, 0.000001 not 1e-06, and 6.02e23 not 602000…000."""
    if value and not 1e-6 <= abs(value) < 1e15:
        return f"{value:.15g}".replace("e+", "e")
    return format(Decimal(repr(value)).normalize(), "f")


def format_fraction(value: float) -> str:
    """``value`` as a fraction is written: ½ for 0.5, and as `format_number` otherwise."""
    return _FRACTIONS.get(value) or format_number(value)


def format_rounded(value: float) -> str:
    """``value`` to two decimal places, as marks are written: 0.33 not 0.3333…"""
    return format_number(round(value, 2))


def sentence(text: str) -> str:
    """``text`` ending with a full stop, unless its last word or quoted value already ends one."""
    return text if text.rstrip('`”" ').endswith((".", "?", "!")) else f"{text}."


def plural(noun: str, count: int) -> str:
    """``noun`` as it reads after ``count``: “cell” after 1, “cells” after any other number."""
    return noun if count == 1 else f"{noun}s"


def capitalised(text: str) -> str:
    """``text`` starting with a capital letter, the rest as it is."""
    return text[:1].upper() + text[1:]


def plain_text(markdown: str) -> str:
    """``markdown`` with its escapes undone and its LaTeX formulas typed out: ``$\\frac{1}{2}$`` as
    ``1/2``."""
    return _ESCAPE_OR_FORMULA.sub(
        lambda match: match[1] or latex_text(match[2] or match[3]), markdown
    )


def plain_markdown(markdown: str) -> str:
    """``markdown`` as plain text, as a description reads in feedback: code in quotes, bold undone,
    and escapes and formulas as ``plain_text`` writes them."""
    return plain_text(_CODE_SPAN.sub(lambda match: quoted(_code_text(match[2])), markdown)).replace(
        "**", ""
    )


def _code_text(text: str) -> str:
    """A code span's text, without the space each side that ``markdown_code`` pads it with."""
    return text[1:-1] if text.startswith(" ") and text.endswith(" ") and text.strip() else text


def as_text(value: object) -> str:
    """``value`` as text, with floats written as people write them."""
    return format_number(value) if isinstance(value, float) else str(value)


def _join(items: list[str], conjunction: Literal["and", "or"]) -> str:
    if len(items) <= 2:
        return f" {conjunction} ".join(items)
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def quoted(value: object) -> str:
    """``value`` in curly double quotes, for plain-text feedback, without a final full stop: the
    sentence around it has its own."""
    return f"“{as_text(value).removesuffix('.')}”"


def plain_join(values: Iterable[object], *, conjunction: Literal["and", "or"]) -> str:
    return _join([as_text(value) for value in values], conjunction)


def quoted_join(values: Iterable[object], *, conjunction: Literal["and", "or"]) -> str:
    return _join([quoted(value) for value in values], conjunction)


def markdown_code(value: object) -> str:
    text = as_text(value)
    fence = "`" * (max(map(len, _BACKTICK_RUN.findall(text)), default=0) + 1)
    if text.startswith(("`", " ")) or text.endswith(("`", " ")):
        text = f" {text} "
    return f"{fence}{text}{fence}"


def fenced(code: str, language: str = "python") -> str:
    """``code`` as a Markdown code block, fenced longer than any run of backticks in it."""
    fence = "`" * max([3, *(len(run) + 1 for run in _BACKTICK_RUN.findall(code))])
    return f"{fence}{language}\n{code.rstrip()}\n{fence}"


def markdown_join(values: Iterable[object], *, conjunction: Literal["and", "or"]) -> str:
    return _join([markdown_code(value) for value in values], conjunction)


def markdown_list(items: Iterable[str]) -> str:
    """``items`` as a Markdown bullet list, each item's later lines under it."""
    return "\n".join(f"- {textwrap.indent(item, '  ').removeprefix('  ')}" for item in items)
