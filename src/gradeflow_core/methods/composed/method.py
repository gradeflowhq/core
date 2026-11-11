"""A composed method: its file's steps expanded with a step's settings, run as one."""

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ...common.paths import Location, is_within, json_pointer, location_of
from ...common.problems import InvalidError, Problem
from ...questions.base import Question
from ...questions.types import ALL_QUESTION_TYPES, QuestionId
from ...steps.context import Context
from ...steps.method import Method
from ...steps.outcome import (
    Outcome,
    QuestionOutcomes,
    Skipped,
    TransformResult,
    outcome_of,
)
from ...steps.places import output_question, places, usable_places
from ...steps.position import OutputQuestions, Position
from ...steps.problems import place_problems
from ...steps.running import mark, rendered_feedback, transform
from ...steps.step import Step
from ...steps.templates import TemplateError, render
from ...steps.workflow import NO_STEPS_PROBLEM
from .declared_settings import settings_model
from .expand import ComposedSettings, file_scope
from .file import MethodFile, methods_used_by_file
from .validation import validate_templates


class ComposedMethod(Method[ComposedSettings]):
    """A method defined by a method file."""

    def __init__(self, name: str, file: MethodFile, methods_by_ref: Mapping[str, Method]) -> None:
        self.name = name
        self.version = file.version
        self.file = file
        self.label = file.label
        self.summary = file.summary
        self.guide = file.guide
        self.level = file.level
        self.question_types = (
            ALL_QUESTION_TYPES if file.question_types == "all" else frozenset(file.question_types)
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
            setting_name: deepcopy(declared_setting.default)
            for setting_name, declared_setting in self.file.settings.items()
            if isinstance(declared_setting.default, list) and declared_setting.default
        }

    def gives_marks(self, settings: Mapping[str, Any]) -> bool:
        return self.file.output == "marks"

    def runs_inner_steps_in_order(self, settings: Mapping[str, Any]) -> bool:
        return True

    def describe(self, settings: ComposedSettings) -> str:
        expansion = settings.expansion
        if self.file.description is None:
            return expansion.describe()
        step_descriptions = [
            inner_step.describe()
            for inner_step in expansion.resolution.inner_steps.values()
            if inner_step is not None
        ]
        return render(self.file.description, {**file_scope(settings), "parts": step_descriptions})

    def questions_marked(
        self, settings: ComposedSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        expansion = _expansion_or_none(settings)
        if expansion is None or expansion.weight == 0:
            return set()
        return expansion.questions_marked(question_id)

    def questions_read(
        self, settings: ComposedSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        expansion = _expansion_or_none(settings)
        return expansion.questions_read(question_id) if expansion else set()

    def output_question_at(
        self,
        settings: ComposedSettings,
        position: Position,
        inner_output_questions: OutputQuestions,
    ) -> Question | None:
        expansion = _expansion_or_none(settings)
        return output_question(expansion, position) if expansion else None

    def problems(self, settings: ComposedSettings, position: Position) -> Iterator[Problem]:
        """What is wrong with the steps its file expands to, as problems of the step as a whole.
        The steps written in its settings have their own."""
        try:
            expansion = settings.expansion
        except InvalidError as error:
            # Name the settings at fault, not where the file's steps put them.
            yield Problem(f"{self.label}: {' '.join(map(str, error.problems))}")
            return
        except TemplateError as error:
            yield Problem(f"{self.label}: {error}")
            return
        expansion_places = places(expansion, position)
        settings_steps = [
            (
                place.location,
                ("with", *settings_location),
                settings_location[0] in settings.model_fields_set,
            )
            for place in expansion_places
            for settings_location, settings_step in _steps_in_settings(settings)
            if place.step is settings_step
        ]

        def settings_step_containing(location: Location) -> tuple[Location, Location, bool] | None:
            """The step the settings hold that ``location`` is in: where it is in the expansion
            and in the settings, and whether the settings write it."""
            return next((step for step in settings_steps if is_within(location, step[0])), None)

        # Steps written in the settings report their own problems.
        places_to_ask = [
            place
            for place in expansion_places
            if not ((step := settings_step_containing(place.location)) and step[2])
        ]
        for problem in place_problems(usable_places(places_to_ask)):
            location = location_of(problem.step_path)
            match settings_step_containing(location):
                # A step a setting holds by default has its problems where the setting holds it.
                case (expansion_location, settings_location, False):
                    inner_location = location[len(expansion_location) :]
                    step_path = json_pointer((*settings_location, *inner_location))
                    yield replace(problem, step_path=step_path)
                # A list an optional setting fills may be left with no steps.
                case _ if replace(problem, step_path="") != NO_STEPS_PROBLEM:
                    yield Problem(problem.message, is_incomplete=problem.is_incomplete)

    def mark_students(
        self,
        settings: ComposedSettings,
        contexts: Sequence[Context],
        question_id: QuestionId | None,
    ) -> list[QuestionOutcomes]:
        scope = file_scope(settings)
        return [
            self._with_file_feedback_and_labels(scope, outcomes, context)
            for outcomes, context in zip(
                mark(settings.expansion, contexts, question_id), contexts, strict=True
            )
        ]

    def transform_students(
        self,
        settings: ComposedSettings,
        contexts: Sequence[Context],
        question_id: QuestionId | None,
    ) -> list[TransformResult | Skipped]:
        return [
            result.transform_result
            for result in transform(settings.expansion, contexts, question_id)
        ]

    def _with_file_feedback_and_labels(
        self, scope: Mapping[str, Any], results: QuestionOutcomes, context: Context
    ) -> dict[QuestionId, Outcome | Skipped]:
        template = self.file.feedback
        # A method file with a description is titled by it, not by the steps it is written as.
        has_description = self.file.description is not None
        results_with_feedback: dict[QuestionId, Outcome | Skipped] = {}
        for question_id, result in results.items():
            if template is not None:
                result = replace(
                    result,
                    feedback=rendered_feedback(
                        template, scope, outcome_of(result), context, question_id
                    ),
                )
            if has_description and isinstance(result, Outcome):
                result = replace(result, title="", labels=())
            results_with_feedback[question_id] = result
        return results_with_feedback


def _expansion_or_none(settings: ComposedSettings) -> Step | None:
    """The workflow a step with ``settings`` is; None if they make none, which its problems
    say."""
    try:
        return settings.expansion
    except InvalidError, TemplateError:
        return None


def _steps_in_settings(settings: ComposedSettings) -> Iterator[tuple[Location, Step]]:
    """Each step the settings hold, written or by default, and where: ``("sets", 0)``."""
    for name, value in file_scope(settings)["settings"].items():
        if isinstance(value, Step):
            yield (name,), value
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Step):
                    yield (name, index), item
