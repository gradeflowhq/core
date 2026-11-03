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
from ..questions.reading import ListReading
from ..questions.text import TextQuestion
from ..questions.types import QuestionId, ValueType
from ..questions.values import BLANK_SEPARATOR, ValuesQuestion
from .errors import file_read_errors
from .mathml import mathml_to_latex
from .rtf import ParagraphPart, paragraphs

_QUESTION = re.compile(r"Question #:\s*(\d+)")
_ITEM_FIELD = re.compile(r"Item (\w+):\s*(.*)")
_CHOICE = re.compile(r"✓?([A-Z])\.(?:\s.*)?")
_ACCEPTED_ANSWERS_LINE = re.compile(r"(\d+)\.(?:\s(.*))?")
_BLANK = re.compile(r"\xa0+(\d+)\xa0+")
_UNDERLINE_RULE = re.compile(r"_{10,}")
_INLINE_MARKUP = re.compile(r"[\\`*_$]|\[(?=.*\][(\[:])|<(?=[^<>\s]*>)|&(?=#?\w+;)")
"""What Markdown would read as formatting, a link, an autolink or an entity, wherever it is."""
_BLOCK_MARKUP = re.compile(r"\d{1,9}(?=[.)](?:\s|$))|(?=[#>+=~-])")
"""What Markdown would read as a heading, list, quote, rule or fence at a paragraph's start."""
_CLOSING_PUNCTUATION = ".,;:?!)]"
_PNG = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class _DocumentQuestion:
    description: str
    max_marks: float | None
    choices: tuple[str, ...]
    """The choices' letters, if it has choices."""
    correct_choice_count: int
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
    items = _document_questions(data)
    return QuestionSet(
        {
            key: _question(item, question)
            for key, question in questions.items()
            if (item := items.get(key)) is not None
        }
    )


def _question(item: _DocumentQuestion, existing: Question) -> Question:
    common_fields = {
        "description": item.description,
        "max_marks": existing.max_marks if item.max_marks is None else item.max_marks,
    }
    if item.choices:
        allows_multiple = isinstance(existing, ChoiceQuestion) and existing.allow_multiple
        return ChoiceQuestion.model_validate(
            {
                **common_fields,
                "options": item.choices,
                "allow_multiple": allows_multiple or item.correct_choice_count > 1,
            }
        )
    if not item.blanks:
        return existing.model_copy(update=common_fields)
    value_types: list[ValueType] = [
        "number" if _all_numbers(blank) else "text" for blank in item.blanks
    ]
    if len(value_types) > 1:
        return ValuesQuestion.model_validate(
            {
                **common_fields,
                "reading": ListReading(separator=BLANK_SEPARATOR),
                "value_types": value_types,
            }
        )
    return (NumberQuestion if value_types == ["number"] else TextQuestion).model_validate(
        common_fields
    )


def _all_numbers(accepted: tuple[str, ...]) -> bool:
    """Whether every accepted answer is a number."""
    return bool(accepted) and all(map(reads_as_number, accepted))


@file_read_errors()
def _document_questions(data: bytes) -> dict[QuestionId, _DocumentQuestion]:
    lines_by_question: dict[QuestionId, list[str]] = {}
    if data.startswith(b"{\\rtf"):
        current_lines: list[str] | None = None
        for line in _lines(data.decode("latin-1")):
            if question := _QUESTION.fullmatch(line):
                current_lines = lines_by_question.setdefault(f"Q{question[1]}", [])
            elif current_lines is not None:
                current_lines.append(line)
    if not lines_by_question:
        raise GradeFlowError("Couldn't read the file: it isn't an ExamSoft question document.")
    return {key: _document_question(lines) for key, lines in lines_by_question.items()}


def _document_question(lines: list[str]) -> _DocumentQuestion:
    fields_start = next(
        (i for i, line in enumerate(lines) if _ITEM_FIELD.fullmatch(line)), len(lines)
    )
    fields = {
        match[1]: match[2]
        for line in lines[fields_start:]
        if (match := _ITEM_FIELD.fullmatch(line))
    }
    stem_lines, blanks = _split_accepted_answers(lines[:fields_start])
    choices_start = None
    if not blanks:
        choices_start = next(
            (i for i in reversed(range(fields_start)) if _choice_letter(stem_lines[i]) == "A"), None
        )
    choices: tuple[str, ...] = ()
    correct_choice_count = 0
    if choices_start is not None:
        choices = tuple(
            letter for line in stem_lines[choices_start:] if (letter := _choice_letter(line))
        )
        correct_choice_count = sum(line.startswith("✓") for line in stem_lines[choices_start:])
        stem_lines = [
            *stem_lines[:choices_start],
            *(line.removeprefix("✓") for line in stem_lines[choices_start:]),
        ]
    description = "\n\n".join(map(_escaped_paragraph, stem_lines))
    return _DocumentQuestion(
        description, _float_or_none(fields.get("Weight", "")), choices, correct_choice_count, blanks
    )


def _escaped_paragraph(line: str) -> str:
    """``line`` as a Markdown paragraph that reads as it is."""
    markup = _BLOCK_MARKUP.match(line)
    return line if markup is None else f"{markup[0]}\\{line[markup.end() :]}"


def _choice_letter(line: str) -> str | None:
    return match[1] if (match := _CHOICE.fullmatch(line)) else None


def _float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _split_accepted_answers(lines: list[str]) -> tuple[list[str], tuple[tuple[str, ...], ...]]:
    """``lines`` without the accepted answers listed after a stem with blanks (1) to (n), and
    each blank's accepted answers, escaped as Markdown like the rest."""
    last_match = _ACCEPTED_ANSWERS_LINE.fullmatch(lines[-1]) if lines else None
    blank_count = int(last_match[1]) if last_match else 0
    if not 0 < blank_count < len(lines) or f"({blank_count})" not in "\n".join(
        lines[:-blank_count]
    ):
        return lines, ()
    matches = [_ACCEPTED_ANSWERS_LINE.fullmatch(line) for line in lines[-blank_count:]]
    if [int(match[1]) if match else 0 for match in matches] != list(range(1, blank_count + 1)):
        return lines, ()
    blanks = tuple(
        tuple(answer for answer in (match[2] or "").split("|") if answer.strip())
        for match in matches
        if match
    )
    return lines[:-blank_count], blanks


def _lines(text: str) -> list[str]:
    """The document's non-blank paragraphs, with formulas and blanks joined to the text around."""
    lines: list[str] = []
    previous_is_inline = False
    for parts in paragraphs(text):
        line, is_inline = _markdown_line(parts)
        if not line:
            continue
        joins_previous = is_inline or (previous_is_inline and not _starts_entry(line))
        if joins_previous and lines and not _starts_entry(lines[-1], (_QUESTION, _ITEM_FIELD)):
            lines[-1] += line if line[0] in _CLOSING_PUNCTUATION else f" {line}"
        else:
            lines.append(line)
        previous_is_inline = is_inline
    return lines


def _markdown_line(parts: list[ParagraphPart]) -> tuple[str, bool]:
    """A paragraph's Markdown and whether it joins the text around it; nothing for a rule."""
    text = [part for part in parts if isinstance(part, str)]
    if len(text) == len(parts) and _UNDERLINE_RULE.fullmatch("".join(text).strip()):
        return "", False
    markdown = "".join(
        _INLINE_MARKUP.sub(r"\\\g<0>", part) if isinstance(part, str) else _formula(part)
        for part in parts
    )
    if blank := _BLANK.fullmatch(markdown.strip(" \t")):
        return f"({blank[1]})", True
    line = " ".join(markdown.split())
    return line, bool(line) and all(isinstance(part, bytes) or part.isspace() for part in parts)


def _starts_entry(
    line: str,
    patterns: tuple[re.Pattern[str], ...] = (
        _CHOICE,
        _ITEM_FIELD,
        _QUESTION,
        _ACCEPTED_ANSWERS_LINE,
    ),
) -> bool:
    return any(pattern.fullmatch(line) for pattern in patterns)


def _formula(picture: bytes) -> str:
    """The MathML a PNG carries, as LaTeX between ``$`` signs; blank for other pictures."""
    if not picture.startswith(_PNG):
        return ""
    position = len(_PNG)
    while position + 8 <= len(picture):
        length, chunk_type = struct.unpack_from(">I4s", picture, position)
        keyword, _, text = picture[position + 8 : position + 8 + length].partition(b"\0")
        if chunk_type == b"tEXt" and keyword == b"MathML":
            try:
                latex = mathml_to_latex(text.decode("utf-8", "replace"))
            except ParseError, RecursionError:
                return ""
            return f"${latex}$" if latex else ""
        position += 12 + length
    return ""
