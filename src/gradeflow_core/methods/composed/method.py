"""A composed method: its file's checks expanded with a check's inputs, marked as one."""

from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any

from pydantic import PrivateAttr

from ...common.paths import Location
from ...questions.types import ALL_TYPES, QuestionId
from ..check import Check, Context, rendered_feedback
from ..method import Method
from ..outcome import Outcome
from ..position import Position
from ..settings import Settings
from ..templates import render
from .expand import expand
from .file import MethodFile, methods_used_by_file
from .inputs import settings_model
from .validation import validate_templates


class ComposedCheck(Check):
    _expanded_check: Check | None = PrivateAttr(default=None)
    """The Combination it is, with its inputs filled in, once worked out."""

    def expanded(self) -> Check:
        """The Combination it is, with its inputs filled in."""
        if self._expanded_check is None:
            assert isinstance(self.method, ComposedMethod)  # its model is a composed method's
            scope = {"inputs": dict(self.with_)}
            self._expanded_check = expand(self.method.file.combination(), scope, self._methods)
        return self._expanded_check


class ComposedMethod(Method[Settings]):
    _check_base_class = ComposedCheck

    def __init__(
        self, name: str, file: MethodFile, methods_by_ref: Mapping[str, Method[Any]]
    ) -> None:
        self.name = name
        self.version = file.version
        self.file = file
        self.label = file.label
        self.summary = file.summary
        self.guide = file.guide
        self.level = file.level
        self.question_types = (
            ALL_TYPES if file.question_types == "all" else frozenset(file.question_types)
        )
        self.marks_question_group = file.marks_question_group
        self.examples = tuple(file.examples)
        self.settings = settings_model(file, methods_by_ref)
        self.needs_model_endpoint = any(
            methods_by_ref[ref].needs_model_endpoint
            for ref in methods_used_by_file(file)
            if ref in methods_by_ref
        )
        validate_templates(file, self.settings)

    def _expanded(self, check: Check) -> Check:
        assert isinstance(check, ComposedCheck)  # validated with this method's model
        return check.expanded()

    def initial_settings(self, position: Position) -> dict[str, Any]:
        """Lists a new check starts with, such as Answer Sets' first set."""
        return {
            key: declared_input.default
            for key, declared_input in self.file.inputs.items()
            if isinstance(declared_input.default, list) and declared_input.default
        }

    def describe(self, check: Check) -> str:
        expanded = self._expanded(check)
        if self.file.description is None:
            return expanded.describe()
        parts = [part.describe() for _, part, _ in expanded.method.inner_checks(expanded)]
        return render(
            self.file.description, {"inputs": dict(self._settings_of(check)), "parts": parts}
        )

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        outcomes = self._expanded(check).mark(context, question, value)
        template = self.file.feedback
        scope = {"inputs": dict(self._settings_of(check))}
        # A method file with a description is titled by it, not by the checks it is written as.
        has_description = self.file.description is not None
        return {
            key: replace(
                outcome,
                method=self.label,
                feedback=outcome.feedback
                if template is None
                else rendered_feedback(template, scope, outcome, context, key, value),
                title="" if has_description else outcome.title,
                labels=() if has_description else outcome.labels,
            )
            for key, outcome in outcomes.items()
        }

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return self._expanded(check).questions_marked(question)

    def questions_read(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return self._expanded(check).questions_read(question)

    def expanded_checks(self, check: Check) -> Iterator[tuple[Location, Check, bool]]:
        yield ("generated", self.name), self._expanded(check), False
