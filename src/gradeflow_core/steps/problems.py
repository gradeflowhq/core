"""What stops a marking from marking."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from typing import Any

from ..common.paths import is_within
from ..common.problems import Problem
from .places import StepPlace, places, usable_places
from .position import Position
from .step import STEP_PROPERTIES, OutcomeFeedback, Step, method_and_settings
from .templates import CHECK_RESULT_SHAPE, STEP_RESULT_SHAPE, Shape, template_errors


def located_problems(
    marking: Step,
    position: Position,
    path: str = "",
    *,
    scope: Mapping[str, Shape] = {},
) -> list[Problem]:
    """What stops ``marking``, at ``path``, from marking inside ``position``, each where it
    is: what is wrong with its data, then with what its steps read and do. A step that can't be
    where it is has only that problem. Its templates may also read ``scope``."""
    step_places = usable_places(places(marking, position, path))
    usable_paths = {place.path for place in step_places if place.position_problem is None}
    data_problems = (
        replace(problem, step_path=path + problem.step_path)
        for problem in marking.resolution.data_problems
    )
    problems = [
        *(problem for problem in data_problems if problem.step_path in usable_paths),
        *place_problems(step_places, scope),
    ]
    if marking.question is not None and not marking.resolution.data_problems:
        other_question_ids = (marking.questions_read() | marking.questions_marked()) - {
            marking.question
        }
        if other_question_ids:
            question_ids_text = ", ".join(sorted(other_question_ids))
            problems.append(
                Problem(f"Use a group marking to read or award {question_ids_text}.", path)
            )
    return problems


def place_problems(
    step_places: Iterable[StepPlace], scope: Mapping[str, Shape] = {}
) -> Iterator[Problem]:
    """What stops the steps at ``step_places`` marking where they are, besides what is wrong with
    their data."""
    for place in step_places:
        step, step_path = place.step, place.path
        if step is None:
            continue
        if place.shares_id:
            yield Problem(f"Another step here is called {step.id}.", step_path, "/id")
        if place.position_problem is not None:
            yield replace(place.position_problem, step_path=step_path)
            continue
        if (
            step.resolution.gives_marks
            and step.weight != STEP_PROPERTIES["weight"].default
            and not place.position.is_in_list
        ):
            yield Problem("Only a step in a list has a weight.", step_path, "/weight")
        # A method file's templates were checked when it loaded.
        if not step.resolution.scope:
            for field, error in template_errors(
                _templates(step), {**scope, "steps": _shapes(place)}
            ):
                yield Problem(str(error), step_path, field)
        if not step.resolution.data_problems:
            method, settings = method_and_settings(step)
            for problem in method.problems(settings, place.position):
                yield _method_problem_in_step(problem, step, step_path)


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


def _method_problem_in_step(problem: Problem, step: Step, step_path: str) -> Problem:
    """``problem``, which the method of ``step`` at ``step_path`` found in its settings, at the
    step written in them that its field is in, or else at that field of ``step``."""
    if problem.step_path or not problem.field:
        return replace(problem, step_path=step_path + problem.step_path)
    field = "/with" + problem.field
    for path_in_step, inner_step in step.resolution.inner_steps.items():
        if inner_step is not None and is_within(field, path_in_step):
            return replace(
                problem, step_path=step_path + path_in_step, field=field[len(path_in_step) :]
            )
    return replace(problem, step_path=step_path, field=field)
