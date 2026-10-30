from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .problems import Problem


class GradeFlowError(Exception):
    pass


class InvalidError(GradeFlowError):
    """Data that isn't valid: ``problems`` says what is wrong, each where it is."""

    def __init__(self, problems: list[Problem]) -> None:
        self.problems = problems
        located = (f"{p.where}: {p.message}" if p.where else p.message for p in problems)
        super().__init__("\n".join(located))


class EditingError(GradeFlowError):
    """A request to edit a marking names a path or method that doesn't fit where it is."""


class MethodFileError(GradeFlowError):
    """A method's YAML that doesn't define a method, said at the line at fault."""
