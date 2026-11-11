"""A problem: something wrong, in words for people, where it is and on which line."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import ValidationError
from pydantic_core import ErrorDetails, PydanticCustomError

from .error_formatting import error_message
from .exceptions import GradeFlowError
from .paths import is_list_item, last_part, parent, pointer

_INCOMPLETE_ERROR_TYPES = frozenset({"missing", "too_short"})
"""What pydantic finds of a value not filled in yet, rather than wrong."""
_PROBLEM_ERROR_TYPE = "problem"


@dataclass(frozen=True)
class Problem:
    """Something wrong, in words for people, and the step and field at fault, if any."""

    message: str
    step_path: str = ""
    """Where the step is, a JSON Pointer such as ``/with/steps/0``; ``""`` is the marking itself."""
    field: str = ""
    """Where in the step, a JSON Pointer such as ``/with/answers/0``."""
    line: int | None = None
    """The line at fault, from 1, of the YAML it was read from."""
    is_incomplete: bool = False
    """Not filled in yet, rather than wrong."""

    @property
    def where(self) -> str:
        """The step and field at fault, such as ``/markings/0/with/answers/1``."""
        return self.step_path + self.field

    def __str__(self) -> str:
        """The message after its line and the field at fault, such as ``Line 4: answers: Add at
        least one.``"""
        field_name = _field_name(self.field)
        text = (
            f"{field_name}: {self.message}"
            if field_name not in ("", "with", "uses")
            else self.message
        )
        return with_line(self.line, text)


def _field_name(field: str) -> str:
    """The name of the field at ``field``, past any list index."""
    while is_list_item(field):
        field = parent(field)
    return last_part(field)


def with_line(line: int | None, text: str) -> str:
    """``text`` after its ``line``, if it has one."""
    return f"Line {line}: {text}" if line is not None else text


class InvalidError(GradeFlowError):
    """Data that isn't valid: ``problems`` says what is wrong, each where it is."""

    def __init__(self, problems: list[Problem]) -> None:
        self.problems = problems
        messages = (
            with_line(
                problem.line,
                f"{problem.where}: {problem.message}" if problem.where else problem.message,
            )
            for problem in problems
        )
        super().__init__("\n".join(messages))


@contextmanager
def invalid_errors(data: object) -> Generator[None]:
    """Raises ``InvalidError`` with the problems pydantic finds validating ``data`` inside."""
    try:
        yield
    except ValidationError as error:
        raise InvalidError(validation_problems(error, data)) from None


def validation_problems(error: ValidationError, data: object) -> list[Problem]:
    """The problems pydantic found validating ``data``, each at its field."""
    problems = (validation_problem(error_details, data) for error_details in error.errors())
    return list(dict.fromkeys(problems))


def validation_problem(error_details: ErrorDetails, data: object) -> Problem:
    """A problem pydantic found validating ``data``, at its field."""
    error_type = error_details["type"]
    field = _pointer_in(data, error_details["loc"], ends_at_missing_key=error_type == "missing")
    if error_type == _PROBLEM_ERROR_TYPE:
        field += error_details["ctx"]["where"]
    return Problem(
        error_message(error_details),
        field=field,
        is_incomplete=error_type in _INCOMPLETE_ERROR_TYPES,
    )


def problem_error(problem: Problem) -> PydanticCustomError:
    """``problem`` as the error a pydantic validator raises for the value it is in, which
    `validation_problem` reads back where it is."""
    context = {"message": problem.message, "where": problem.where}
    return PydanticCustomError(_PROBLEM_ERROR_TYPE, "{message}", context)


def _pointer_in(data: object, location: tuple[str | int, ...], *, ends_at_missing_key: bool) -> str:
    """The JSON Pointer into ``data`` of ``location``, without the parts pydantic adds that
    aren't keys of the data, such as the member of a union it tried."""
    parts: list[str | int] = []
    value = data
    for index, part in enumerate(location, start=1):
        items = (
            value
            if isinstance(value, Mapping)
            else dict(enumerate(value))
            if isinstance(value, list | tuple)
            else {}
        )
        if part in items:
            value = items[part]
        elif not (ends_at_missing_key and index == len(location)):
            continue
        parts.append(part)
    return pointer(*parts)
