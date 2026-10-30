"""
ExamSoft's question document: an assessment's Print/Export as DOC, which is RTF.

Each question comes as ``Question #: n``, its stem, its choices (``A. text``, with ``✓`` before
correct ones) or the accepted answers of its blanks (``1. a|b|``), then its ``Item …`` fields, of
which ``Item Weight`` is its marks. Formulas are PNG pictures carrying their MathML in a text
chunk, each in a paragraph of its own, and blanks are underlined numbers, also on their own.

A question's text becomes Markdown, its formulas LaTeX between ``$`` signs.
"""

import re
import struct
from dataclasses import dataclass
from xml.etree.ElementTree import ParseError

from ..common.exceptions import GradeFlowError
from ..questions.base import Question
from ..questions.choice import ChoiceQuestion
from ..questions.number import NumberQuestion, reads_as_number
from ..questions.question_set import QuestionSet
from ..questions.text import TextQuestion
from ..questions.types import QuestionId, ValueType
from ..questions.values import ValuesQuestion, values_reading
from .errors import adapter_errors
from .mathml import mathml_latex
from .rtf import Part, paragraphs

_QUESTION = re.compile(r"Question #:\s*(\d+)")
_FIELD = re.compile(r"Item (\w+):\s*(.*)")
_CHOICE = re.compile(r"✓?([A-Z])\.(?:\s.*)?")
_ACCEPTED = re.compile(r"(\d+)\.(?:\s(.*))?")
_BLANK = re.compile(r"\xa0+(\d+)\xa0+")
_RULE = re.compile(r"_{10,}")
_INLINE_MARK = re.compile(r"[\\`*_$]|\[(?=.*\][(\[:])|<(?=[^<>\s]*>)|&(?=#?\w+;)")
"""What Markdown would read as formatting, a link, an autolink or an entity, wherever it is."""
_BLOCK_MARK = re.compile(r"\d{1,9}(?=[.)](?:\s|$))|(?=[#>+=~-])")
"""What Markdown would read as a heading, list, quote, rule or fence at a paragraph's start."""
_CLOSING = ".,;:?!)]"
_PNG = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class _Item:
    description: str
    max_marks: float | None
    choices: tuple[str, ...]
    """The choices' letters, if it has choices."""
    ticked: int
    """How many choices are marked correct."""
    blanks: tuple[tuple[str, ...], ...]
    """Each blank's accepted answers, if it has blanks."""


def read_question_document(data: bytes, questions: QuestionSet) -> QuestionSet:
    """
    The ``questions`` an ExamSoft question document has, set up as it shows them: choices make a
    choice question, one blank a number or text question, several a multiple-values question.
    Its text is their description and its weight their maximum marks. A question with neither
    choices nor blanks keeps its type, and a choice question that allows several options still
    does.
    """
    items = _items(data)
    return QuestionSet(
        {
            key: _question(item, question)
            for key, question in questions.items()
            if (item := items.get(key)) is not None
        }
    )


def _question(item: _Item, current: Question) -> Question:
    shared = {
        "description": item.description,
        "max_marks": current.max_marks if item.max_marks is None else item.max_marks,
    }
    if item.choices:
        several = isinstance(current, ChoiceQuestion) and current.allow_multiple
        return ChoiceQuestion.model_validate(
            {**shared, "options": item.choices, "allow_multiple": several or item.ticked > 1}
        )
    if not item.blanks:
        return current.model_copy(update=shared)
    types: list[ValueType] = ["number" if _numbers(blank) else "text" for blank in item.blanks]
    if len(types) > 1:
        return ValuesQuestion.model_validate(
            {**shared, "reading": values_reading(), "value_types": types}
        )
    return (NumberQuestion if types == ["number"] else TextQuestion).model_validate(shared)


def _numbers(accepted: tuple[str, ...]) -> bool:
    """Whether every accepted answer is a number."""
    return bool(accepted) and all(map(reads_as_number, accepted))


@adapter_errors()
def _items(data: bytes) -> dict[QuestionId, _Item]:
    blocks: dict[QuestionId, list[str]] = {}
    if data.startswith(b"{\\rtf"):
        current: list[str] | None = None
        for line in _lines(data.decode("latin-1")):
            if question := _QUESTION.fullmatch(line):
                current = blocks.setdefault(f"Q{question[1]}", [])
            elif current is not None:
                current.append(line)
    if not blocks:
        raise GradeFlowError("Couldn't read the file: it isn't an ExamSoft question document.")
    return {key: _item(lines) for key, lines in blocks.items()}


def _item(lines: list[str]) -> _Item:
    end = next((i for i, line in enumerate(lines) if _FIELD.fullmatch(line)), len(lines))
    fields = {match[1]: match[2] for line in lines[end:] if (match := _FIELD.fullmatch(line))}
    content, blanks = _accepted(lines[:end])
    start = None
    if not blanks:
        start = next((i for i in reversed(range(end)) if _choice(content[i]) == "A"), None)
    choices: tuple[str, ...] = ()
    ticked = 0
    if start is not None:
        choices = tuple(letter for line in content[start:] if (letter := _choice(line)))
        ticked = sum(line.startswith("✓") for line in content[start:])
        content = [*content[:start], *(line.removeprefix("✓") for line in content[start:])]
    description = "\n\n".join(map(_paragraph, content))
    return _Item(description, _number(fields.get("Weight", "")), choices, ticked, blanks)


def _paragraph(line: str) -> str:
    """``line`` as a Markdown paragraph that reads as it is."""
    mark = _BLOCK_MARK.match(line)
    return line if mark is None else f"{mark[0]}\\{line[mark.end() :]}"


def _choice(line: str) -> str | None:
    return match[1] if (match := _CHOICE.fullmatch(line)) else None


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _accepted(lines: list[str]) -> tuple[list[str], tuple[tuple[str, ...], ...]]:
    """``lines`` without the accepted answers listed after a stem with blanks (1) to (n), and
    each blank's accepted answers, escaped as Markdown like the rest."""
    last = _ACCEPTED.fullmatch(lines[-1]) if lines else None
    count = int(last[1]) if last else 0
    if not 0 < count < len(lines) or f"({count})" not in "\n".join(lines[:-count]):
        return lines, ()
    matches = [_ACCEPTED.fullmatch(line) for line in lines[-count:]]
    if [int(match[1]) if match else 0 for match in matches] != list(range(1, count + 1)):
        return lines, ()
    blanks = tuple(
        tuple(answer for answer in (match[2] or "").split("|") if answer.strip())
        for match in matches
        if match
    )
    return lines[:-count], blanks


def _lines(text: str) -> list[str]:
    """The document's non-blank paragraphs, with formulas and blanks joined to the text around."""
    lines: list[str] = []
    attach = False
    for parts in paragraphs(text):
        line, inline = _line(parts)
        if not line:
            continue
        joins = inline or (attach and not _starts_entry(line))
        if joins and lines and not _starts_entry(lines[-1], (_QUESTION, _FIELD)):
            lines[-1] += line if line[0] in _CLOSING else f" {line}"
        else:
            lines.append(line)
        attach = inline
    return lines


def _line(parts: list[Part]) -> tuple[str, bool]:
    """A paragraph's Markdown and whether it joins the text around it; nothing for a rule."""
    text = [part for part in parts if isinstance(part, str)]
    if len(text) == len(parts) and _RULE.fullmatch("".join(text).strip()):
        return "", False
    raw = "".join(
        _INLINE_MARK.sub(r"\\\g<0>", part) if isinstance(part, str) else _formula(part)
        for part in parts
    )
    if blank := _BLANK.fullmatch(raw.strip(" \t")):
        return f"({blank[1]})", True
    line = " ".join(raw.split())
    return line, bool(line) and all(isinstance(part, bytes) or part.isspace() for part in parts)


def _starts_entry(
    line: str, entries: tuple[re.Pattern[str], ...] = (_CHOICE, _FIELD, _QUESTION, _ACCEPTED)
) -> bool:
    return any(entry.fullmatch(line) for entry in entries)


def _formula(picture: bytes) -> str:
    """The MathML a PNG carries, as LaTeX between ``$`` signs; blank for other pictures."""
    if not picture.startswith(_PNG):
        return ""
    position = len(_PNG)
    while position + 8 <= len(picture):
        length, kind = struct.unpack_from(">I4s", picture, position)
        name, _, value = picture[position + 8 : position + 8 + length].partition(b"\0")
        if kind == b"tEXt" and name == b"MathML":
            try:
                latex = mathml_latex(value.decode("utf-8", "replace"))
            except ParseError, RecursionError:
                return ""
            return f"${latex}$" if latex else ""
        position += 12 + length
    return ""
