"""Questions worked out from their answers."""

import re
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator

from ..common.python_layouts import guess_layouts
from . import new_question
from .base import Question
from .choice import ChoiceQuestion
from .code import CodeQuestion
from .code_files import split_answer
from .list import ItemType, ListQuestion
from .multipart import MultipartQuestion
from .number import NumberQuestion, reads_as_number
from .parts import numbered_parts, unbracketed
from .reading import (
    EXAMSOFT_SEPARATOR,
    ListReading,
    Reading,
    is_blank,
    read_code,
    read_list,
    sorted_options,
)
from .text import TextQuestion
from .types import QuestionType, ValueType

MAX_CHOICE_OPTIONS = 7
"""Most distinct values a question may have to be worked out as a choice."""
SEPARATORS = (",", ";", "|", "/")
"""What else may separate an answer's values, after the answer key's own."""


def infer_question(answers: list[str]) -> Question:
    if multipart_question := _same_count_multipart_question(answers, (EXAMSOFT_SEPARATOR,)):
        return multipart_question
    if _are_mostly_numbers(answers):
        return NumberQuestion()
    if _looks_like_choices(answers):
        return _choice_question(answers)
    if _are_mostly_code(answers):
        return CodeQuestion()
    multipart_question = _same_count_multipart_question(answers, SEPARATORS)
    return (
        multipart_question
        if multipart_question and all(part.type == "number" for part in multipart_question.parts)
        else TextQuestion()
    )


def converted_question(
    question: Question, question_type: QuestionType, answers: list[str]
) -> Question:
    """``question`` as the chosen type, keeping what every question has, with what the type needs
    read from its answers: a choice's options; a multi-part question's separator, number of parts
    and their types; or a list's separator and item type. Without answers to read them from, a
    choice has the options of a new one."""
    common_fields = {name: getattr(question, name) for name in Question.model_fields}
    if question_type == "choice" and _distinct_values(answers, ListReading()):
        question_of_type: Question = _choice_question(answers)
    elif question_type == "multipart":
        question_of_type = _multipart_question(answers)
    elif question_type == "list":
        question_of_type = _list_question(answers)
    else:
        question_of_type = new_question(question_type)
    return question_of_type.model_copy(update=common_fields)


def _same_count_multipart_question(
    answers: list[str], separators: Iterable[str]
) -> MultipartQuestion | None:
    """Split on the first separator giving every answer the same number of values, more than one."""
    for separator in separators:
        reading = ListReading(separator=separator)
        value_counts = {
            _value_count(values, separator)
            for values in _value_lists_inside_brackets(answers, reading)
        }
        if len(value_counts) == 1 and (value_count := next(iter(value_counts))) > 1:
            return MultipartQuestion(
                reading=reading, parts=numbered_parts(_value_types(answers, reading, value_count))
            )
    return None


def _value_count(values: list[str], separator: str) -> int:
    """How many values an answer has: every blank with ExamSoft's separator, which writes empty
    ones too; only those written with any other."""
    return len(values) if separator == EXAMSOFT_SEPARATOR else len(list(filter(None, values)))


def _reading_that_splits_most(answers: list[str]) -> ListReading:
    """A reading with the first separator, ExamSoft's then the others, that splits most answers;
    the default if none does."""
    for separator in (EXAMSOFT_SEPARATOR, *SEPARATORS):
        reading = ListReading(separator=separator)
        value_lists = list(_value_lists_inside_brackets(answers, reading))
        if sum(len(values) > 1 for values in value_lists) > len(value_lists) / 2:
            return reading
    return ListReading()


def _value_lists(answers: list[str], reading: ListReading) -> Iterator[list[str]]:
    return (values for answer in answers if (values := read_list(answer, reading)) is not None)


def _value_lists_inside_brackets(answers: list[str], reading: ListReading) -> Iterator[list[str]]:
    """The values of each answer that isn't blank, read inside brackets around them as multi-part
    and list questions read them."""
    return _value_lists([unbracketed(answer) for answer in answers], reading)


def _distinct_values(answers: list[str], reading: ListReading) -> set[str]:
    return {value for values in _value_lists(answers, reading) for value in values if value}


def _are_mostly_numbers(texts: list[str]) -> bool:
    """Whether most answered ``texts`` read as numbers."""
    answered_texts = [text for text in texts if not is_blank(text.strip(), Reading())]
    return sum(map(reads_as_number, answered_texts)) > len(answered_texts) / 2


def _are_mostly_code(answers: list[str]) -> bool:
    """Whether most answers are programs: several lines, in the file that runs, that some layout
    compiles. A single word compiles too, as a name, and prose rarely does."""
    main_files = (next(iter(split_answer(answer).values())) for answer in answers)
    code_answers = [
        code for source in main_files if (code := read_code(source, Reading())) is not None
    ]
    program_count = sum("\n" in code and bool(guess_layouts(code, 1)) for code in code_answers)
    return program_count > len(code_answers) / 2


def _value_types(answers: list[str], reading: ListReading, value_count: int) -> list[ValueType]:
    """Each value is a number when most answers with ``value_count`` values give a number there."""
    full_value_lists = [
        values
        for values in _value_lists_inside_brackets(answers, reading)
        if len(values) == value_count
    ]
    return [
        "number" if _are_mostly_numbers([values[i] for values in full_value_lists]) else "text"
        for i in range(value_count)
    ]


_OPTION_PATTERN = re.compile(r"[^\W\d_]+(?:[ '-][^\W\d_]+)*|\d")
"""What an option looks like: words, such as `B`, `true` or `active transport`, or a digit;
not `H2O`, `-2` or a date, which are answers written out."""


def _looks_like_choices(answers: list[str]) -> bool:
    """Whether the answers choose among a few options that look like options, one of them
    chosen in more than one answer: students share options, while unlike answers differ."""
    option_sets = [set(filter(None, values)) for values in _value_lists(answers, ListReading())]
    option_counts = Counter(option for options in option_sets for option in options)
    return (
        0 < len(option_counts) <= MAX_CHOICE_OPTIONS
        and max(option_counts.values()) > 1
        and all(map(_OPTION_PATTERN.fullmatch, option_counts))
    )


def _choice_question(answers: list[str]) -> ChoiceQuestion:
    reading = ListReading()
    return ChoiceQuestion(
        options=tuple(sorted_options(_distinct_values(answers, reading))),
        allow_multiple={len(values) for values in _value_lists(answers, reading)} != {1},
    )


def _multipart_question(answers: list[str]) -> MultipartQuestion:
    """Read with the separator that splits the answers: as many parts as answers most often have
    values, and at least two."""
    reading = _reading_that_splits_most(answers)
    value_counts = [
        len(values) for values in _value_lists_inside_brackets(answers, reading) if len(values) > 1
    ]
    value_count = statistics.mode(value_counts) if value_counts else 2
    return MultipartQuestion(
        reading=reading, parts=numbered_parts(_value_types(answers, reading, value_count))
    )


def _list_question(answers: list[str]) -> ListQuestion:
    """Read with the separator that splits the answers: of numbers when most values are
    numbers."""
    reading = _reading_that_splits_most(answers)
    values = [
        value for values in _value_lists_inside_brackets(answers, reading) for value in values
    ]
    item: ItemType = "number" if _are_mostly_numbers(values) else "text"
    return ListQuestion(reading=reading, item=item)
