"""Questions worked out from their answers."""

import re
from collections import Counter
from collections.abc import Iterable, Iterator

from . import new_question
from .base import Question
from .choice import ChoiceQuestion
from .code import CodeQuestion
from .layouts import guess_layouts
from .number import NumberQuestion, reads_as_number
from .reading import ListReading, Reading, is_blank, read_code, read_list
from .text import TextQuestion
from .types import QuestionType, ValueType
from .values import EXAMSOFT_SEPARATOR, ValuesQuestion

CHOICE_OPTION_LIMIT = 7
"""Most distinct values a question may have to be worked out as a choice."""
SEPARATORS = (",", ";", "|", "/")
"""What else may separate an answer's values, after the answer key's own."""


def infer_question(answers: list[str]) -> Question:
    if values := _same_count_values_question(answers, (EXAMSOFT_SEPARATOR,)):
        return values
    if _mostly_numbers(answers):
        return NumberQuestion()
    if _looks_like_choices(answers):
        return _choice_question(answers)
    if _mostly_code(answers):
        return CodeQuestion()
    values = _same_count_values_question(answers, SEPARATORS)
    return (
        values
        if values and all(value_type == "number" for value_type in values.value_types)
        else TextQuestion()
    )


def converted_question(
    question: Question, question_type: QuestionType, answers: list[str]
) -> Question:
    """``question`` as the chosen type, keeping what every question has, with what the type needs
    read from its answers: a choice's options, or how many values an answer has between the
    default separator and what each is. Without answers to read them from, a choice has the
    options of a new one."""
    common_fields = {name: getattr(question, name) for name in Question.model_fields}
    return _question_of_type(question_type, answers).model_copy(update=common_fields)


def _question_of_type(question_type: QuestionType, answers: list[str]) -> Question:
    match question_type:
        case "choice":
            return (
                _choice_question(answers)
                if _distinct_values(answers, ListReading())
                else new_question("choice")
            )
        case "values":
            return _values_question(answers)
    return new_question(question_type)


def _same_count_values_question(
    answers: list[str], separators: Iterable[str]
) -> ValuesQuestion | None:
    """Split on the first separator giving every answer the same number of values, more than one."""
    for separator in separators:
        reading = ListReading(separator=separator)
        counts = {len(list(filter(None, values))) for values in _split_answers(answers, reading)}
        if len(counts) == 1 and (value_count := next(iter(counts))) > 1:
            return ValuesQuestion(
                reading=reading, value_types=_value_types(answers, reading, value_count)
            )
    return None


def _split_answers(answers: list[str], reading: ListReading) -> Iterator[list[str]]:
    return (values for answer in answers if (values := read_list(answer, reading)) is not None)


def _value_counts(answers: list[str], reading: ListReading) -> set[int]:
    return {len(values) for values in _split_answers(answers, reading)}


def _distinct_values(answers: list[str], reading: ListReading) -> set[str]:
    return {value for values in _split_answers(answers, reading) for value in values if value}


def _mostly_numbers(texts: list[str]) -> bool:
    """Whether most answered ``texts`` read as numbers."""
    answered = [text for text in texts if not is_blank(text.strip(), Reading())]
    return sum(map(reads_as_number, answered)) > len(answered) / 2


def _mostly_code(answers: list[str]) -> bool:
    """Whether most answers are programs: several lines that some layout compiles. A single word
    compiles too, as a name, and prose rarely does."""
    code_answers = [
        code for answer in answers if (code := read_code(answer, Reading())) is not None
    ]
    program_count = sum("\n" in code and bool(guess_layouts(code, 1)) for code in code_answers)
    return program_count > len(code_answers) / 2


def _value_types(answers: list[str], reading: ListReading, value_count: int) -> list[ValueType]:
    """Each value is a number when most answers with ``value_count`` values give a number there."""
    with_value_count = [
        values for values in _split_answers(answers, reading) if len(values) == value_count
    ]
    return [
        "number" if _mostly_numbers([values[i] for values in with_value_count]) else "text"
        for i in range(value_count)
    ]


_OPTION_PATTERN = re.compile(r"[^\W\d_]+(?:[ '-][^\W\d_]+)*|\d")
"""What an option looks like: words, such as `B`, `true` or `active transport`, or a digit;
not `H2O`, `-2` or a date, which are answers written out."""


def _is_option(value: str) -> bool:
    return _OPTION_PATTERN.fullmatch(value) is not None


def _looks_like_choices(answers: list[str]) -> bool:
    """Whether the answers choose among a few options that look like options, one of them
    chosen in more than one answer: students share options, while unlike answers differ."""
    option_sets = [set(filter(None, values)) for values in _split_answers(answers, ListReading())]
    option_counts = Counter(option for options in option_sets for option in options)
    return (
        0 < len(option_counts) <= CHOICE_OPTION_LIMIT
        and max(option_counts.values()) > 1
        and all(map(_is_option, option_counts))
    )


def _choice_question(answers: list[str]) -> ChoiceQuestion:
    reading = ListReading()
    return ChoiceQuestion(
        options=frozenset(_distinct_values(answers, reading)),
        allow_multiple=_value_counts(answers, reading) != {1},
    )


def _values_question(answers: list[str]) -> ValuesQuestion:
    """Read with the default separator, which a person may change: as many values as answers
    most often have, and at least two."""
    reading = ListReading()
    value_counts = Counter(
        value_count
        for values in _split_answers(answers, reading)
        if (value_count := len(values)) > 1
    )
    value_count = value_counts.most_common(1)[0][0] if value_counts else 2
    return ValuesQuestion(reading=reading, value_types=_value_types(answers, reading, value_count))
