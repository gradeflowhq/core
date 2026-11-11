"""What stops a marking from marking."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from typing import Any

from ..common.paths import Location, is_within, json_pointer, location_of
from ..common.problems import Problem
from .places import StepPlace, places, usable_places
from .position import Position
from .step import OutcomeFeedback, Step, method_and_settings
from .templates import CHECK_RESULT_SHAPE, STEP_RESULT_SHAPE, Shape, template_errors


def located_problems(
    marking: Step,
    position: Position,
    location: Location = (),
    *,
    scope: Mapping[str, Shape] = {},
) -> list[Problem]:
    """What stops ``marking``, at ``location``, from marking inside ``position``, each where it
    is: what is wrong with its data, then with what its steps read and do. A step that can't be
    where it is has only that problem. Its templates may also read ``scope``."""
    step_places = usable_places(places(marking, position, location))
    usable_locations = {place.location for place in step_places if place.position_problem is None}
    data_problems = (
        replace(problem, step_path=json_pointer(location) + problem.step_path)
        for problem in marking.resolution.data_problems
    )
    problems = [
        *(
            problem
            for problem in data_problems
            if location_of(problem.step_path) in usable_locations
        ),
        *place_problems(step_places, scope),
    ]
    if marking.question is not None and not marking.resolution.data_problems:
        other_question_ids = (marking.questions_read() | marking.questions_marked()) - {
            marking.question
        }
        if other_question_ids:
            question_ids_text = ", ".join(sorted(other_question_ids))
            problems.append(
                Problem(
                    f"Use a group marking to read or award {question_ids_text}.",
                    json_pointer(location),
                )
            )
    return problems


def place_problems(
    step_places: Iterable[StepPlace], scope: Mapping[str, Shape] = {}
) -> Iterator[Problem]:
    """What stops the steps at ``step_places`` marking where they are, besides what is wrong with
    their data."""
    for place in step_places:
        step, step_path = place.step, json_pointer(place.location)
        if step is None:
            continue
        if place.shares_id:
            yield Problem(f"Another step here is called {step.id}.", step_path, ("id",))
        if place.position_problem is not None:
            yield replace(place.position_problem, step_path=step_path)
            continue
        # A method file's templates were checked when it loaded.
        if not step.resolution.scope:
            for field, error in template_errors(
                _templates(step), {**scope, "steps": _shapes(place)}
            ):
                yield Problem(str(error), step_path, field)
        if not step.resolution.data_problems:
            method, settings = method_and_settings(step)
            for problem in method.problems(settings, place.position):
                yield _at_inner_step(problem, step, place.location)


def _templates(step: Step) -> dict[str, Any]:
    """The step's condition and feedback, as YAML holds them."""
    feedback = step.feedback
    return {
        "if": step.if_,
        "feedback": feedback.model_dump() if isinstance(feedback, OutcomeFeedback) else feedback,
    }


def _shapes(place: StepPlace) -> dict[str, Shape]:
    """What templates see of each earlier step the place can read."""
    return {
        step_id: CHECK_RESULT_SHAPE if step_output_question is None else STEP_RESULT_SHAPE
        for step_id, step_output_question in place.output_questions.items()
    }


def _at_inner_step(problem: Problem, step: Step, location: Location) -> Problem:
    """``problem``, found in the settings of ``step`` at ``location``, at the step written in
    them that it is about, if it is about one."""
    for inner_location, inner_step in step.resolution.inner_steps.items():
        if inner_step is not None and is_within(problem.field, inner_location):
            return replace(
                problem,
                step_path=json_pointer((*location, "with", *inner_location)),
                field=problem.field[len(inner_location) :],
            )
    return replace(problem, step_path=json_pointer(location) + problem.step_path)
