"""Questions worked out from their answers."""

import re
from collections import Counter
from collections.abc import Iterable, Iterator

from . import new_question
from .base import Question
from .choice import ChoiceQuestion, choice_reading
from .code import CodeQuestion
from .layouts import guess_layouts
from .number import NumberQuestion, reads_as_number
from .reading import CodeReading, ListReading, read_code, read_list
from .text import TextQuestion
from .types import QuestionType, ValueType
from .values import BLANK_SEPARATOR, ValuesQuestion

CHOICE_OPTION_LIMIT = 7
"""Most distinct values a question may have to be worked out as a choice."""
SEPARATORS = (BLANK_SEPARATOR, ",", ";", "|", "/")
"""What may separate an answer's values, the answer key's own first."""


def infer_question(answers: list[str]) -> Question:
    if values := _same_count_values(answers, SEPARATORS[:1]):
        return values
    if _mostly_numbers(answers):
        return NumberQuestion()
    if _chosen(answers):
        return _choice(answers)
    if _mostly_code(answers):
        return CodeQuestion()
    values = _same_count_values(answers, SEPARATORS[1:])
    return (
        values
        if values and all(kind == "number" for kind in values.value_types)
        else TextQuestion()
    )


def infer_question_as(
    question: Question, question_type: QuestionType, answers: list[str]
) -> Question:
    """``question`` as the chosen type, keeping what every question has, with what the type needs
    read from its answers: a choice's options, or how many values an answer has between the
    default separator and what each is. Without answers to read them from, a choice has the
    options of a new one."""
    shared = {name: getattr(question, name) for name in Question.model_fields}
    return _question_of_type(question_type, answers).model_copy(update=shared)


def _question_of_type(question_type: QuestionType, answers: list[str]) -> Question:
    match question_type:
        case "choice":
            return (
                _choice(answers)
                if _seen_values(answers, choice_reading())
                else new_question("choice")
            )
        case "values":
            return _values(answers)
    return new_question(question_type)


def _same_count_values(answers: list[str], separators: Iterable[str]) -> ValuesQuestion | None:
    """Split on the first separator giving every answer the same number of values, more than one."""
    for separator in separators:
        reading = ListReading(separator=separator)
        counts = {len(list(filter(None, values))) for values in _lists(answers, reading)}
        if len(counts) == 1 and (size := next(iter(counts))) > 1:
            return ValuesQuestion(reading=reading, value_types=_value_types(answers, reading, size))
    return None


def _lists(answers: list[str], reading: ListReading) -> Iterator[list[str]]:
    return (values for answer in answers if (values := read_list(answer, reading)) is not None)


def _value_counts(answers: list[str], reading: ListReading) -> set[int]:
    return {len(values) for values in _lists(answers, reading)}


def _seen_values(answers: list[str], reading: ListReading) -> set[str]:
    return {value for values in _lists(answers, reading) for value in values if value}


def _mostly_numbers(texts: list[str]) -> bool:
    return sum(map(reads_as_number, texts)) > len(texts) / 2


def _mostly_code(answers: list[str]) -> bool:
    """Whether most answers are programs: several lines that some layout compiles. A single word
    compiles too, as a name, and prose rarely does."""
    codes = [
        code for answer in answers if isinstance(code := read_code(answer, CodeReading()), str)
    ]
    programs = sum("\n" in code and bool(guess_layouts(code, 1)) for code in codes)
    return programs > len(codes) / 2


def _value_types(answers: list[str], reading: ListReading, size: int) -> list[ValueType]:
    """Each value is a number when most answers with ``size`` values give a number there."""
    sized = [values for values in _lists(answers, reading) if len(values) == size]
    return [
        "number" if _mostly_numbers([values[i] for values in sized if values[i]]) else "text"
        for i in range(size)
    ]


_OPTION = re.compile(r"[^\W\d_]+(?:[ '-][^\W\d_]+)*|\d")
"""What an option looks like: words, such as `B`, `true` or `active transport`, or a digit;
not `H2O`, `-2` or a date, which are answers written out."""


def _is_option(value: str) -> bool:
    return _OPTION.fullmatch(value) is not None


def _chosen(answers: list[str]) -> bool:
    """Whether the answers choose among a few options that look like options, one of them
    chosen in more than one answer: students share options, while unlike answers differ."""
    sets = [set(filter(None, values)) for values in _lists(answers, choice_reading())]
    chosen = Counter(option for options in sets for option in options)
    return (
        0 < len(chosen) <= CHOICE_OPTION_LIMIT
        and max(chosen.values()) > 1
        and all(map(_is_option, chosen))
    )


def _choice(answers: list[str]) -> ChoiceQuestion:
    reading = choice_reading()
    return ChoiceQuestion(
        options=frozenset(_seen_values(answers, reading)),
        allow_multiple=_value_counts(answers, reading) != {1},
    )


def _values(answers: list[str]) -> ValuesQuestion:
    """Read with the default separator, which a person may change: as many values as answers
    most often have, and at least two."""
    reading = ListReading()
    sizes = Counter(size for values in _lists(answers, reading) if (size := len(values)) > 1)
    size = sizes.most_common(1)[0][0] if sizes else 2
    return ValuesQuestion(reading=reading, value_types=_value_types(answers, reading, size))
