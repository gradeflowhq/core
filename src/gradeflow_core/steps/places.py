"""Where each step of a marking is and what it reads there."""

from collections.abc import Iterable
from dataclasses import dataclass

from ..common.paths import Location, is_within
from ..common.problems import Problem
from ..questions.base import Question
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from .position import OutputQuestions, Position
from .step import Step


@dataclass(frozen=True)
class StepPlace:
    """Where a step of a marking is, or one can be added, and what a step there reads."""

    location: Location
    step: Step | None
    position: Position
    enclosing_position: Position
    """Where it is before its own question and input: what the steps holding it chose."""
    earlier_locations: tuple[Location, ...]
    """Where the steps before it in its list are, when they run in order."""
    output_questions: OutputQuestions
    """What it can read of the steps before it."""
    position_problem: Problem | None = None
    """Why the step can't be here, if it can't."""
    shares_id: bool = False
    """Whether another step it could be mistaken for has its ID."""


def marking_position(questions: QuestionSet, question_id: QuestionId | None) -> Position:
    """Where a marking of ``question_id`` is, or of a question group without one."""
    return Position(questions, question_id, is_in_group=question_id is None)


def places(
    step: Step | None, enclosing_position: Position, location: Location = ()
) -> list[StepPlace]:
    """The place of ``step`` at ``location``, then each place inside it, in order."""
    return _places_and_output_question(step, enclosing_position, location, (), {})[0]


def usable_places(step_places: Iterable[StepPlace]) -> list[StepPlace]:
    """``step_places``, in the order ``places`` gives them, without those inside a step that
    can't be where it is."""
    usable: list[StepPlace] = []
    unusable_location: Location | None = None
    for place in step_places:
        if unusable_location is not None and is_within(place.location, unusable_location):
            continue
        usable.append(place)
        if place.position_problem is not None:
            unusable_location = place.location
    return usable


def output_question(step: Step, enclosing_position: Position) -> Question | None:
    """The question that reads what ``step`` gives inside ``enclosing_position``; None for a
    check."""
    return _places_and_output_question(step, enclosing_position, (), (), {})[1]


def _places_and_output_question(
    step: Step | None,
    enclosing_position: Position,
    location: Location,
    earlier_locations: tuple[Location, ...],
    output_questions: OutputQuestions,
    *,
    shares_id: bool = False,
) -> tuple[list[StepPlace], Question | None]:
    if step is None:
        empty_place = StepPlace(
            location,
            None,
            enclosing_position,
            enclosing_position,
            earlier_locations,
            output_questions,
        )
        return [empty_place], None
    position, position_problem = _step_position(step, enclosing_position, output_questions)
    step_places = [
        StepPlace(
            location,
            step,
            position,
            enclosing_position,
            earlier_locations,
            output_questions,
            position_problem,
            shares_id,
        )
    ]
    resolution = step.resolution
    if resolution.method is None:
        return step_places, None
    inner_output_questions: dict[str, Question | None] = {}
    known_by_list: dict[Location, dict[str, Question | None]] = {}
    earlier_by_list: dict[Location, tuple[Location, ...]] = {}
    for settings_location, inner_step in resolution.inner_steps.items():
        inner_location = (*location, "with", *settings_location)
        is_in_list = isinstance(settings_location[-1], int)
        list_location = inner_location[:-1] if is_in_list else inner_location
        known = known_by_list.setdefault(list_location, dict(output_questions))
        inner_places, inner_output_question = _places_and_output_question(
            inner_step,
            resolution.method.inner_position(position, is_in_list=is_in_list),
            inner_location,
            earlier_by_list.get(list_location, ()),
            dict(known) if resolution.runs_inner_steps_in_order else output_questions,
            shares_id=is_in_list and inner_step is not None and inner_step.id in known,
        )
        step_places += inner_places
        if inner_step is None:
            continue
        if resolution.runs_inner_steps_in_order:
            earlier_by_list[list_location] = (
                *earlier_by_list.get(list_location, ()),
                inner_location,
            )
        if inner_step.id is not None:
            known[inner_step.id] = inner_output_questions[inner_step.id] = inner_output_question
    if step.question is not None and step.question not in enclosing_position.questions:
        # A question that isn't there is ignored when working out what it gives.
        position = enclosing_position.reading(step.source, output_questions)[0]
    return step_places, _output_question(step, position, inner_output_questions)


def _output_question(
    step: Step, position: Position, inner_output_questions: OutputQuestions
) -> Question | None:
    """The question that reads what ``step`` gives when it reads at ``position``, given what the
    steps in its settings give; None for a check, or a step that reads no question. Until its
    settings are valid, a transform is taken to give what it reads."""
    method = step.resolution.method
    if step.resolution.gives_marks or method is None or position.question_model is None:
        return None
    if step.settings is None:
        return position.question_model
    return method.output_question_at(step.settings, position, inner_output_questions)


def _step_position(
    step: Step, enclosing_position: Position, output_questions: OutputQuestions
) -> tuple[Position, Problem | None]:
    """Where the step reads, and what stops it marking there. A step of no known method is only
    placed."""
    step_position = enclosing_position.for_question(step.question)
    method = step.resolution.method
    if method is None:
        return step_position.reading(step.source, output_questions)[0], None
    if step.question is not None and step.question not in enclosing_position.questions:
        unknown_question_message = f"{step.question} isn't one of the questions."
        return step_position, Problem(unknown_question_message, field=("question",))
    input_position, refusal = step_position.reading(step.source, output_questions)
    if refusal:
        return step_position, Problem(refusal, field=("input",))
    needs_question = input_position.question is None and input_position.input_kind == "output"
    if needs_question or (
        input_position.question_model is None and not method.marks_question_group
    ):
        return input_position, Problem("Choose the question it checks.", field=("question",))
    if not method.can_run_at(input_position):
        message = f"{method.label} can't be used on {_position_kind(input_position)}."
        return input_position, Problem(message)
    return input_position, None


def _position_kind(position: Position) -> str:
    question = position.question_model
    if question is None:
        return "a question group"
    if position.input_kind is not None:
        return f"a {question.type} {position.input_kind}"
    return f"a {question.label.lower()} question"
