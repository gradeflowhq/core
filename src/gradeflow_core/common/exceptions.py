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
        messages = (
            f"{problem.where}: {problem.message}" if problem.where else problem.message
            for problem in problems
        )
        super().__init__("\n".join(messages))


class EditingError(GradeFlowError):
    """A request to edit a marking names a path or method that doesn't fit where it is."""


class MethodFileError(GradeFlowError):
    """A method's YAML that doesn't define a method, with the line at fault. ``messages`` lists
    every problem found, the first being its message."""

    def __init__(self, message: str, *others: str) -> None:
        super().__init__(message)
        self.messages = [message, *others]
