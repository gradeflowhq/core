"""A composed method: its file's steps expanded with a step's settings, run as one."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

from pydantic import ValidationError

from ...common.exceptions import InvalidError
from ...common.paths import Location
from ...questions.base import Question
from ...questions.types import ALL_TYPES, QuestionId
from ..method import Method
from ..outcome import MethodResult, Outcome
from ..position import Position
from ..settings import Settings
from ..step import Check, Context, Step, Transform, load_step, rendered_feedback
from ..templates import TemplateError, render
from .declared_settings import settings_model
from .expand import expand_step
from .file import MethodFile, methods_used_by_file
from .validation import validate_templates


class ComposedMethod(Method[Settings]):
    """A method defined by a method file."""

    _holds_steps = True  # whatever its settings, it runs through the steps its file expands to

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

    def initial_settings(self, position: Position) -> dict[str, Any]:
        """Lists a new step starts with, such as Answer Sets' first set."""
        return {
            key: declared_setting.default
            for key, declared_setting in self.file.settings.items()
            if isinstance(declared_setting.default, list) and declared_setting.default
        }

    def gives_marks(self, settings: Mapping[str, Any]) -> bool:
        return self.file.output == "marks"

    def _expansion_of(self, step: Step) -> Step:
        """The workflow ``step`` is, with its settings filled in."""
        if step._cached_expansion is None:  # noqa: SLF001
            scope = {"settings": dict(step.with_)}
            workflow = self.file.workflow()
            step._cached_expansion = expand_step(workflow, scope, step._methods)  # noqa: SLF001
        return step._cached_expansion  # noqa: SLF001

    def describe(self, step: Step) -> str:
        expansion = self._expansion_of(step)
        if self.file.description is None:
            return expansion.describe()
        parts = [part.describe() for _, part in expansion.method.inner_steps(expansion)]
        return render(
            self.file.description, {"settings": dict(self._settings_of(step)), "parts": parts}
        )

    def run(
        self, step: Step, contexts: Sequence[Context], question: QuestionId | None
    ) -> list[MethodResult]:
        expansion = self._expansion_of(step)
        if isinstance(expansion, Transform):
            return [result.transform_result for result in expansion.transform(contexts, question)]
        outcomes_by_student = cast(Check, expansion).mark(contexts, question)
        return [
            self._with_file_feedback_and_labels(cast(Check, step), outcomes, context)
            for outcomes, context in zip(outcomes_by_student, contexts, strict=True)
        ]

    def _with_file_feedback_and_labels(
        self, check: Check, outcomes: Mapping[QuestionId, Outcome], context: Context
    ) -> dict[QuestionId, Outcome]:
        template = self.file.feedback
        scope = {"settings": dict(self._settings_of(check))}
        # A method file with a description is titled by it, not by the steps it is written as.
        has_description = self.file.description is not None
        return {
            key: replace(
                outcome,
                method=self.label,
                feedback=outcome.feedback
                if template is None
                else rendered_feedback(template, scope, outcome, context, key),
                title="" if has_description else outcome.title,
                labels=() if has_description else outcome.labels,
            )
            for key, outcome in outcomes.items()
        }

    def output_question(
        self,
        settings: Mapping[str, Any],
        source_question: Question,
        methods: Mapping[str, Method[Any]],
    ) -> Question | None:
        if self.gives_marks(settings):
            return None
        try:
            step = load_step({"uses": self.ref, "with": settings}, methods)
            workflow = self._expansion_of(step).dump()
        except ValidationError, InvalidError, TemplateError:  # settings still being written
            workflow = self.file.workflow()
        return methods[workflow["uses"]].output_question(workflow["with"], source_question, methods)

    def reads_earlier_results(self, settings: Mapping[str, Any]) -> bool:
        return True  # its file's steps run in order, as a workflow's do

    def expanded_steps(self, step: Step) -> Iterator[tuple[Location, Step]]:
        yield ("generated", self.name), self._expansion_of(step)
