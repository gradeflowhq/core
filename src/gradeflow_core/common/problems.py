"""A problem: something wrong, in words for people, where it is."""

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from .error_formatting import error_message
from .exceptions import InvalidError
from .paths import Location, get, join

_INCOMPLETE_ERROR_TYPES = frozenset({"missing", "too_short"})
"""What pydantic finds of a value not filled in yet, rather than wrong."""


@dataclass(frozen=True)
class Problem:
    """Something wrong, in words for people: with the check at ``check``, if in a marking, in its
    field at ``field``, if any."""

    message: str
    check: str = ""
    """Where the check is, such as ``with.checks.0``; ``""`` is the marking itself."""
    field: Location = ()
    """Where in the check, its settings and keys as one, such as ``("answers", 0)``."""
    is_incomplete: bool = False
    """Not filled in yet, rather than wrong."""

    @property
    def where(self) -> str:
        """The check and field at fault, such as ``markings.0.with.answers.1``."""
        return join(self.check, *self.field)

    def __str__(self) -> str:
        """The message after the field at fault, such as ``answers: Add at least one.``"""
        field_name = next((part for part in reversed(self.field) if isinstance(part, str)), "")
        return (
            f"{field_name}: {self.message}"
            if field_name not in ("", "with", "uses")
            else self.message
        )


@contextmanager
def invalid_errors(data: object) -> Generator[None]:
    """Raises ``InvalidError`` with the problems pydantic finds validating ``data`` inside."""
    try:
        yield
    except ValidationError as error:
        raise InvalidError(validation_problems(error, data)) from None


def validation_problems(
    error: ValidationError, data: object, is_check: Callable[[Location], bool] | None = None
) -> list[Problem]:
    """The problems pydantic found validating ``data``, each at the check holding it, if any:
    where ``is_check`` says one is, or else a mapping whose ``uses`` names a method, or is at
    fault."""

    def problem_of(details: ErrorDetails) -> Problem:
        location = _untagged(tuple(details["loc"]), data)

        def is_check_at(at: Location) -> bool:
            node = get(data, join(*at))
            return isinstance(node, Mapping) and (
                isinstance(node.get("uses"), str) or location[len(at) :] == ("uses",)
            )

        is_incomplete = details["type"] in _INCOMPLETE_ERROR_TYPES
        problem = at_innermost_check(
            Problem(error_message(details), is_incomplete=is_incomplete),
            location,
            is_check or is_check_at,
        )
        if problem.field == ("with",) and details["type"] == "value_error":
            return replace(problem, field=())  # a rule over all its settings is the check's
        return problem

    return list(dict.fromkeys(map(problem_of, error.errors())))


def _untagged(location: Location, data: object) -> Location:
    """``location`` without the tags pydantic puts in it for a member of a tagged union: a value
    of the object there, such as its ``type``, rather than one of its keys."""
    parts: list[str | int] = []
    node = data
    for part in location:
        if isinstance(node, Mapping) and part not in node and part in node.values():
            continue
        parts.append(part)
        node = get(node, str(part))
    return tuple(parts)


def at_innermost_check(
    problem: Problem, location: Location, is_check: Callable[[Location], bool]
) -> Problem:
    """``problem``, found at ``location`` of the data, at the innermost check holding it; a check's
    settings are its fields, without their ``with``."""
    checks = [length for length in range(len(location) + 1) if is_check(location[:length])]
    if not checks:
        return replace(problem, field=location)
    end, field = checks[-1], location[checks[-1] :]
    return replace(
        problem,
        check=join(*location[:end]),
        field=field[1:] if len(field) > 1 and field[0] == "with" else field,
    )
