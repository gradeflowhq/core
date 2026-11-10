"""
ExamSoft's question document: an assessment's Print/Export as DOC, which is RTF.

Each question comes as ``Question #: n``, its stem, its choices (``A. text``, with ``✓`` before
correct ones) or the accepted answers of its blanks (``1. a|b|``), then its ``Item …`` fields, of
which ``Item Weight`` is its marks. Formulas are PNG pictures carrying their MathML in a text
chunk, each in a paragraph of its own, and blanks are underlined numbers, also on their own.

A question's text becomes Markdown, its formulas LaTeX between ``$`` signs.
"""

import math
import re
import struct
from typing import Any
from xml.etree.ElementTree import ParseError

from ..common.error_formatting import shortened_reason
from ..common.exceptions import GradeFlowError
from ..questions.base import Question
from ..questions.choice import ChoiceQuestion
from ..questions.number import reads_as_number
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId, ValueType
from ..questions.values import EXAMSOFT_SEPARATOR
from .mathml import mathml_to_latex
from .rtf import ParagraphPart, paragraphs

_QUESTION_LINE = re.compile(r"Question #:\s*(\d+)")
_ITEM_FIELD = re.compile(r"Item (\w+):\s*(.*)")
_CHOICE_LINE = re.compile(r"✓?([A-Z])\.(?:\s.*)?")
_ACCEPTED_ANSWERS_LINE = re.compile(r"(\d+)\.(?:\s(.*))?")
_BLANK_NUMBER = re.compile(r"\xa0+(\d+)\xa0+")
_UNDERLINE_RULE = re.compile(r"_{10,}")
_INLINE_MARKUP = re.compile(r"[\\`*_$]|\[(?=.*\][(\[:])|<(?=[^<>\s]*>)|&(?=#?\w+;)")
"""What Markdown would read as formatting, a link, an autolink or an entity, wherever it is."""
_BLOCK_MARKUP = re.compile(r"\d{1,9}(?=[.)](?:\s|$))|(?=[#>+=~-])")
"""What Markdown would read as a heading, list, quote, rule or fence at a paragraph's start."""
_CLOSING_PUNCTUATION = ".,;:?!)]"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_question_document(data: bytes, questions: QuestionSet) -> QuestionSet:
    """
    The ``questions`` an ExamSoft question document has, set up as it shows them: choices make a
    choice question, one blank a number or text question, several a multiple-values question.
    Its text is their description and its weight their maximum marks. A question with neither
    choices nor blanks keeps its type, and a choice question that allows several options still
    does. Raises ``InvalidError`` for a question it can't set up, such as one with a choice twice.
    """
    try:
        lines_by_question = _lines_by_question(data)
        data_by_question = {
            question_id: _question_data(lines, question)
            for question_id, question in questions.items()
            if (lines := lines_by_question.get(question_id)) is not None
        }
    except GradeFlowError:
        raise
    except Exception as error:
        raise GradeFlowError(f"Couldn't read the file: {shortened_reason(str(error))}") from error
    return QuestionSet.load(data_by_question)


def _lines_by_question(data: bytes) -> dict[QuestionId, list[str]]:
    lines_by_question: dict[QuestionId, list[str]] = {}
    if data.startswith(b"{\\rtf"):
        current_lines: list[str] | None = None
        for line in _lines(data.decode("latin-1")):
            if match := _QUESTION_LINE.fullmatch(line):
                current_lines = lines_by_question.setdefault(f"Q{match[1]}", [])
            elif current_lines is not None:
                current_lines.append(line)
    if not lines_by_question:
        raise GradeFlowError("Couldn't read the file: it isn't an ExamSoft question document.")
    return lines_by_question


def _question_data(lines: list[str], current_question: Question) -> dict[str, Any]:
    fields_start = next(
        (i for i, line in enumerate(lines) if _ITEM_FIELD.fullmatch(line)), len(lines)
    )
    fields = {
        match[1]: match[2]
        for line in lines[fields_start:]
        if (match := _ITEM_FIELD.fullmatch(line))
    }
    stem_lines, accepted_answers_by_blank = _split_accepted_answers(lines[:fields_start])
    choices_start = None
    if not accepted_answers_by_blank:
        choices_start = next(
            (i for i in reversed(range(fields_start)) if _choice_letter(stem_lines[i]) == "A"), None
        )
    choice_letters: tuple[str, ...] = ()
    has_several_correct_choices = False
    if choices_start is not None:
        choice_letters = tuple(
            letter for line in stem_lines[choices_start:] if (letter := _choice_letter(line))
        )
        has_several_correct_choices = (
            sum(line.startswith("✓") for line in stem_lines[choices_start:]) > 1
        )
        stem_lines = [
            *stem_lines[:choices_start],
            *(line.removeprefix("✓") for line in stem_lines[choices_start:]),
        ]
    marks = _marks_or_none(fields.get("Weight", ""))
    common_fields = {
        "description": "\n\n".join(map(_escaped_paragraph, stem_lines)),
        "max_marks": current_question.max_marks if marks is None else marks,
    }
    if choice_letters:
        allows_multiple = (
            isinstance(current_question, ChoiceQuestion) and current_question.allow_multiple
        )
        return {
            "type": "choice",
            **common_fields,
            "options": choice_letters,
            "allow_multiple": allows_multiple or has_several_correct_choices,
        }
    if not accepted_answers_by_blank:
        return current_question.dump() | common_fields
    value_types: list[ValueType] = [
        "number" if _are_all_numbers(accepted_answers) else "text"
        for accepted_answers in accepted_answers_by_blank
    ]
    if len(value_types) > 1:
        return {
            "type": "values",
            **common_fields,
            "reading": {"separator": EXAMSOFT_SEPARATOR},
            "value_types": value_types,
        }
    return {"type": "number" if value_types == ["number"] else "text", **common_fields}


def _are_all_numbers(accepted_answers: tuple[str, ...]) -> bool:
    """Whether every accepted answer is a number."""
    return bool(accepted_answers) and all(map(reads_as_number, accepted_answers))


def _escaped_paragraph(line: str) -> str:
    """``line`` as a Markdown paragraph that reads as it is."""
    markup = _BLOCK_MARKUP.match(line)
    return line if markup is None else f"{markup[0]}\\{line[markup.end() :]}"


def _choice_letter(line: str) -> str | None:
    return match[1] if (match := _CHOICE_LINE.fullmatch(line)) else None


def _marks_or_none(text: str) -> float | None:
    try:
        marks = float(text)
    except ValueError:
        return None
    return marks if math.isfinite(marks) and marks >= 0 else None


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
    accepted_answers_by_blank = tuple(
        tuple(answer for answer in (match[2] or "").split("|") if answer.strip())
        for match in matches
        if match
    )
    return lines[:-blank_count], accepted_answers_by_blank


def _lines(text: str) -> list[str]:
    """The document's non-blank paragraphs, with formulas and blanks joined to the text around."""
    lines: list[str] = []
    previous_is_inline = False
    for parts in paragraphs(text):
        line, is_inline = _markdown_line(parts)
        if not line:
            continue
        joins_previous = is_inline or (previous_is_inline and not _starts_entry(line))
        if joins_previous and lines and not _starts_entry(lines[-1], (_QUESTION_LINE, _ITEM_FIELD)):
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
    if match := _BLANK_NUMBER.fullmatch(markdown.strip(" \t")):
        return f"({match[1]})", True
    line = " ".join(markdown.split())
    return line, bool(line) and all(isinstance(part, bytes) or part.isspace() for part in parts)


def _starts_entry(
    line: str,
    patterns: tuple[re.Pattern[str], ...] = (
        _CHOICE_LINE,
        _ITEM_FIELD,
        _QUESTION_LINE,
        _ACCEPTED_ANSWERS_LINE,
    ),
) -> bool:
    return any(pattern.fullmatch(line) for pattern in patterns)


def _formula(picture: bytes) -> str:
    """The MathML a PNG carries, as LaTeX between ``$`` signs; blank for other pictures."""
    if not picture.startswith(_PNG_SIGNATURE):
        return ""
    position = len(_PNG_SIGNATURE)
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
