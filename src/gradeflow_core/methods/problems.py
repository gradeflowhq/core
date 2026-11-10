"""What stops a marking from marking, each where it is."""

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from ..common.exceptions import InvalidError
from ..common.paths import Location, join_path
from ..common.problems import Problem, at_innermost_step
from ..questions.base import Question
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from .input_sources import OutputQuestions, input_source
from .position import Position
from .templates import CHECK_RESULT_SHAPE, STEP_RESULT_SHAPE, Shape, TemplateError, template_errors

if TYPE_CHECKING:
    from .method import Method
    from .step import Step


def located_problems(
    marking: Step,
    questions: QuestionSet,
    location: Location = (),
    *,
    scope: Mapping[str, Shape] = {},
    is_step_at: Callable[[Location], bool] | None = None,
) -> list[Problem]:
    """The problems of a marking at ``location``. Its templates may also read ``scope``."""
    position = Position(
        questions,
        marking.question,
        is_in_group=marking.question is None,
    )
    try:
        locations = {
            id(step): step_location for step_location, step in written_steps(marking, location)
        }
        problems_with_locations = list(
            _problems_with_locations(marking, position, location, location, scope, {}, locations)
        )
        if marking.question is not None:
            other_question_ids = (marking.questions_read() | marking.questions_marked()) - {
                marking.question
            }
            if other_question_ids:
                question_ids_text = ", ".join(sorted(other_question_ids))
                problems_with_locations.append(
                    (
                        location,
                        Problem(f"Use a group marking to read or award {question_ids_text}."),
                    )
                )
    except InvalidError as error:
        # A step generated inside a composed method: name its settings, not where they are written.
        message = " ".join(map(str, error.problems))
        return [Problem(f"{marking.method.label}: {message}", join_path(*location))]
    except TemplateError as error:
        return [Problem(f"{marking.method.label}: {error}", join_path(*location))]
    written_locations = set(locations.values())
    is_step_at = is_step_at or (lambda step_location: step_location in written_locations)
    return [
        at_innermost_step(problem, problem_location, is_step_at)
        for problem_location, problem in problems_with_locations
    ]


def _problems_with_locations(
    step: Step,
    position: Position,
    location: Location,
    written_location: Location,
    scope: Mapping[str, Shape],
    output_questions: OutputQuestions,
    locations: Mapping[int, Location],
) -> Iterator[tuple[Location, Problem]]:
    """Each problem with where it is, as YAML holds the marking. ``written_location`` is where the
    step written in the marking is: the step itself, or the one using the composed method that
    generated it, which has its problems as a whole."""
    if id(step) in locations:
        location = written_location = locations[id(step)]

    def at_written_step(problem_location: Location, problem: Problem) -> tuple[Location, Problem]:
        return (problem_location if location == written_location else written_location), problem

    step_position, position_problem = _step_position(step, position, output_questions)
    if position_problem:
        yield at_written_step((*location, *position_problem.field), position_problem)
        return
    method = step.method
    for problem_location, problem in _template_problems(step, location, scope, output_questions):
        yield at_written_step(problem_location, problem)
    for problem in method.problems(step, step_position):
        yield at_written_step((*location, "with", *problem.field), problem)
    output_questions_by_list: dict[Location, dict[str, Question | None]] = {}
    for inner_location, inner_step in method.expanded_steps(step):
        is_in_list = isinstance(inner_location[-1], int)
        list_location = inner_location[:-1] if is_in_list else inner_location
        sibling_output_questions = output_questions_by_list.setdefault(
            list_location, dict(output_questions)
        )
        if inner_step.id is not None and inner_step.id in sibling_output_questions and is_in_list:
            yield at_written_step(
                (*location, *inner_location, "id"),
                Problem(f"Another step here is called {inner_step.id}."),
            )
        inner_position = method.inner_position(step_position, inner_location)
        readable_output_questions = (
            sibling_output_questions
            if method.reads_earlier_results(dict(step.with_))
            else output_questions
        )
        yield from _problems_with_locations(
            inner_step,
            inner_position,
            (*location, *inner_location),
            written_location,
            scope,
            readable_output_questions,
            locations,
        )
        if inner_step.id is not None:
            sibling_output_questions[inner_step.id] = output_question(
                inner_step.method,
                inner_step.dump().get("with", {}),
                _step_position(inner_step, inner_position, readable_output_questions)[0],
                inner_step._methods,  # noqa: SLF001
            )


def step_input_position(
    question_id: QuestionId | None,
    input_text: object,
    position: Position,
    output_questions: OutputQuestions,
) -> Position:
    """Where a step naming ``question_id`` and reading ``input_text`` reads, inside ``position``; at
    its question alone when ``input_text`` names no input."""
    step_position = position.for_question(question_id)
    try:
        source = input_source(input_text)
    except ValueError:
        return step_position
    return source.input_position(step_position, output_questions)[0]


def output_question(
    method: Method[Any],
    settings: Mapping[str, Any],
    position: Position,
    methods: Mapping[str, Method[Any]],
) -> Question | None:
    """A question reading what a step of ``method`` with ``settings``, as YAML holds them, gives
    when it reads at ``position``; None for a check or a step reading no question."""
    if position.question_model is None:
        return None
    return method.output_question(settings, position, methods)


def _step_position(
    step: Step, position: Position, output_questions: OutputQuestions
) -> tuple[Position, Problem | None]:
    """Where the step reads, and what stops it marking there."""
    if step.question is not None and step.question not in position.questions:
        return position, Problem(
            f"{step.question} isn't one of the questions.", field=("question",)
        )
    step_position = position.for_question(step.question)
    input_position, message = input_source(step.input).input_position(
        step_position, output_questions
    )
    if message:
        return step_position, Problem(message, field=("input",))
    needs_question = input_position.question is None and input_position.input_kind == "output"
    if needs_question or (
        input_position.question_model is None and not step.method.marks_question_group
    ):
        return input_position, Problem("Choose the question it checks.", field=("question",))
    if message := unusable_reason(step.method, input_position):
        return input_position, Problem(message)
    return input_position, None


def written_steps(step: Step, location: Location = ()) -> Iterator[tuple[Location, Step]]:
    """``step`` at ``location`` and the steps written in its settings, each where it is."""
    yield location, step
    for inner_location, inner_step in step.method.inner_steps(step):
        yield from written_steps(inner_step, (*location, *inner_location))


def unusable_reason(method: Method[Any], position: Position) -> str | None:
    """Why ``method`` can't be used at ``position``, if it can't."""
    if method.can_run_at(position):
        return None
    return f"{method.label} can't be used on {_position_kind(position)}."


def _position_kind(position: Position) -> str:
    question = position.question_model
    if question is None:
        return "a question group"
    if position.input_kind is not None:
        return f"a {question.type} {position.input_kind}"
    return f"a {question.label.lower()} question"


def _template_problems(
    step: Step, location: Location, scope: Mapping[str, Shape], output_questions: OutputQuestions
) -> Iterator[tuple[Location, Problem]]:
    if step._scope:  # noqa: SLF001
        return
    earlier_step_shapes = {
        step_id: CHECK_RESULT_SHAPE if step_output_question is None else STEP_RESULT_SHAPE
        for step_id, step_output_question in output_questions.items()
    }
    templates = step.model_dump(mode="json", include={"if_", "feedback"})
    for template_location, error in template_errors(
        templates, {**scope, "steps": earlier_step_shapes}
    ):
        yield (*location, *template_location), Problem(str(error))
