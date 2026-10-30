"""A composed method: its file's checks expanded with a check's inputs, marked as one."""

from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any, Literal

from pydantic import JsonValue, PrivateAttr

from ...common.paths import Loc
from ...questions.types import ALL_TYPES, AnswerShape, QuestionId
from ..check import Check, Context, rendered_feedback
from ..method import Method
from ..outcome import Outcome
from ..position import Position
from ..settings import Settings
from ..templates import render
from .expand import expand
from .file import MethodFile, methods_used
from .inputs import settings_model
from .validation import check_templates


class ComposedCheck(Check):
    _inner: Check | None = PrivateAttr(default=None)
    """The Combination it is, with its inputs filled in, once worked out."""

    def expanded(self) -> Check:
        """The Combination it is, with its inputs filled in."""
        if self._inner is None:
            assert isinstance(self.method, ComposedMethod)  # its model is a composed method's
            scope = {"inputs": dict(self.with_)}
            self._inner = expand(self.method.file.body(), scope, self._methods)
        return self._inner


class ComposedMethod(Method[Settings]):
    _check_base = ComposedCheck

    def __init__(self, name: str, file: MethodFile, lookup: Mapping[str, Method[Any]]) -> None:
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
        # A method for every type passes answers on as they are read; its checks say if they fit.
        self.answer_shape = (
            "both" if file.question_types == "all" else _shared_shape(file.checks, lookup)
        )
        self.group = file.group
        self.examples = tuple(file.examples)
        self.settings = settings_model(file, lookup)
        self.needs_model_endpoint = any(
            lookup[ref].needs_model_endpoint for ref in methods_used(file.body()) if ref in lookup
        )
        check_templates(file, self.settings)

    def _expanded(self, check: Check) -> Check:
        assert isinstance(check, ComposedCheck)  # validated with this method's model
        return check.expanded()

    def initial_settings(self, position: Position) -> dict[str, Any]:
        """Lists a new check starts with, such as Answer Sets' first set."""
        return {
            key: spec.default
            for key, spec in self.file.inputs.items()
            if isinstance(spec.default, list) and spec.default
        }

    def describe(self, check: Check) -> str:
        inner = self._expanded(check)
        if self.file.description is None:
            return inner.describe()
        parts = [part.describe() for _, part, _ in inner.method.nested(inner)]
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
        return {
            key: replace(
                outcome,
                method=self.label,
                feedback=outcome.feedback
                if template is None
                else rendered_feedback(template, scope, outcome, context, key, value),
            )
            for key, outcome in outcomes.items()
        }

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return self._expanded(check).questions_marked(question)

    def questions_read(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return self._expanded(check).questions_read(question)

    def validation_children(self, check: Check) -> Iterator[tuple[Loc, Check, bool]]:
        yield ("generated", self.name), self._expanded(check), False


def _shared_shape(
    checks: list[JsonValue] | str, lookup: Mapping[str, Method[Any]]
) -> AnswerShape | Literal["both"]:
    """What the checks a method file is written as all mark: answers read as several when one
    of them marks only those, as Best Answer does; one answer otherwise."""
    written = checks if isinstance(checks, list) else []
    shapes = {
        lookup[ref].answer_shape
        for check in written
        if isinstance(check, dict) and isinstance(ref := check.get("uses"), str) and ref in lookup
    }
    if shapes == {"both"}:
        return "both"
    return "multiple" if "multiple" in shapes and "single" not in shapes else "single"
