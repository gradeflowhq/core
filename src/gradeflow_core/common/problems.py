"""A problem: something wrong, in words for people, where it is."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from .error_formatting import error_message
from .exceptions import InvalidError
from .paths import Loc, get, join

_INCOMPLETE = frozenset({"missing", "too_short"})
"""What pydantic finds of a value not filled in yet, rather than wrong."""


@dataclass(frozen=True)
class Problem:
    """Something wrong, in words for people: with the check at ``check``, if in a marking, in its
    field at ``field``, if any."""

    message: str
    check: str = ""
    """Where the check is, such as ``with.checks.0``; ``""`` is the marking itself."""
    field: Loc = ()
    """Where in the check, its settings and keys as one, such as ``("answers", 0)``."""
    incomplete: bool = False
    """Not filled in yet, rather than wrong."""

    @property
    def where(self) -> str:
        """The check and field at fault, such as ``markings.0.with.answers.1``."""
        return join(self.check, *self.field)

    def __str__(self) -> str:
        """The message after the field at fault, such as ``answers: Add at least one.``"""
        named = next((part for part in reversed(self.field) if isinstance(part, str)), "")
        return f"{named}: {self.message}" if named not in ("", "with", "uses") else self.message


@contextmanager
def as_invalid(data: object) -> Iterator[None]:
    """Raises ``InvalidError`` with the problems pydantic finds validating ``data`` inside."""
    try:
        yield
    except ValidationError as error:
        raise InvalidError(validation_problems(error, data)) from None


def validation_problems(
    error: ValidationError, data: object, is_check: Callable[[Loc], bool] | None = None
) -> list[Problem]:
    """The problems pydantic found validating ``data``, each at the check holding it, if any:
    where ``is_check`` says one is, or else a mapping whose ``uses`` names a method, or is at
    fault."""

    def problem(details: ErrorDetails) -> Problem:
        loc = _untagged(tuple(details["loc"]), data)

        def holds_uses(at: Loc) -> bool:
            node = get(data, join(*at))
            return isinstance(node, Mapping) and (
                isinstance(node.get("uses"), str) or loc[len(at) :] == ("uses",)
            )

        incomplete = details["type"] in _INCOMPLETE
        found = placed(
            Problem(error_message(details), incomplete=incomplete), loc, is_check or holds_uses
        )
        if found.field == ("with",) and details["type"] == "value_error":
            return replace(found, field=())  # a rule over all its settings is the check's
        return found

    return list(dict.fromkeys(map(problem, error.errors())))


def _untagged(loc: Loc, data: object) -> Loc:
    """``loc`` without the tags pydantic puts in it for a member of a tagged union: a value of the
    object there, such as its ``type``, rather than one of its keys."""
    kept: list[str | int] = []
    node = data
    for part in loc:
        if isinstance(node, Mapping) and part not in node and part in node.values():
            continue
        kept.append(part)
        node = get(node, str(part))
    return tuple(kept)


def placed(problem: Problem, loc: Loc, is_check: Callable[[Loc], bool]) -> Problem:
    """``problem``, found at ``loc`` of the data, at the innermost check holding it; a check's
    settings are its fields, without their ``with``."""
    checks = [n for n in range(len(loc) + 1) if is_check(loc[:n])]
    if not checks:
        return replace(problem, field=loc)
    end, field = checks[-1], loc[checks[-1] :]
    return replace(
        problem,
        check=join(*loc[:end]),
        field=field[1:] if len(field) > 1 and field[0] == "with" else field,
    )
