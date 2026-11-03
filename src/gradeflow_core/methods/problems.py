"""What stops a marking from marking, each where it is."""

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..common.exceptions import InvalidError
from ..common.paths import Location, join
from ..common.problems import Problem, at_innermost_check
from ..questions.question_set import QuestionSet
from .position import Position
from .templates import EARLIER_RESULT, ListOf, Shape, TemplateError, template_errors

if TYPE_CHECKING:
    from .check import Check
    from .method import Method


def located_problems(
    marking: Check,
    questions: QuestionSet,
    location: Location = (),
    *,
    scope: Mapping[str, Shape] = {},
    is_check: Callable[[Location], bool] | None = None,
) -> list[Problem]:
    """The problems of a marking found at ``location``, as in a mark scheme. Its templates may name
    ``scope`` too, as the checks of a method name its inputs. ``is_check`` says where its data has
    a check, when that isn't where the marking has one."""
    position = Position(questions, marking.question, in_group=marking.question is None)
    try:
        locations = {
            id(check): check_location for check_location, check in written_checks(marking, location)
        }
        problems_with_locations = list(
            _problems(marking, position, location, location, scope, {}, locations)
        )
        if marking.question is not None:
            other_questions = (marking.questions_read() | marking.questions_marked()) - {
                marking.question
            }
            if other_questions:
                question_list = ", ".join(sorted(other_questions))
                problems_with_locations.append(
                    (location, Problem(f"Use a group marking to read or award {question_list}."))
                )
    except InvalidError as error:
        # A check generated inside a composed method: name its settings, not where they are written.
        message = " ".join(map(str, error.problems))
        return [Problem(f"{marking.method.label}: {message}", join(*location))]
    except TemplateError as error:
        return [Problem(f"{marking.method.label}: {error}", join(*location))]
    written_locations = set(locations.values())
    is_check = is_check or (lambda check_location: check_location in written_locations)
    return [
        at_innermost_check(problem, problem_location, is_check)
        for problem_location, problem in problems_with_locations
    ]


def _problems(
    check: Check,
    position: Position,
    location: Location,
    written_location: Location,
    scope: Mapping[str, Shape],
    earlier: Mapping[str, Shape],
    locations: Mapping[int, Location],
) -> Iterator[tuple[Location, Problem]]:
    """Each problem with where it is, as YAML holds the marking. ``written_location`` is where the
    check written in the marking is: the check itself, or the one using the composed method that
    generated it, which has its problems as a whole."""
    if id(check) in locations:
        location = written_location = locations[id(check)]

    def at_written_check(problem_location: Location, problem: Problem) -> tuple[Location, Problem]:
        return (problem_location if location == written_location else written_location), problem

    if cannot_mark_reason := _cannot_mark_reason(check, position):
        field, message = cannot_mark_reason
        yield at_written_check((*location, *field), Problem(message))
        return
    method = check.method
    check_position = position.for_check(check.question, check.value)
    for problem_location, problem in _template_problems(check, location, scope, earlier):
        yield at_written_check(problem_location, problem)
    for problem in method.problems(check, check_position):
        yield at_written_check((*location, "with", *problem.field), problem)
    earlier_by_list: dict[Location, dict[str, Any]] = {}
    for inner_location, inner, in_list in method.expanded_checks(check):
        list_location = inner_location[:-1] if in_list else inner_location
        siblings = earlier_by_list.setdefault(list_location, dict(earlier))
        if inner.id is not None and inner.id in siblings and in_list:
            yield at_written_check(
                (*location, *inner_location, "id"),
                Problem(f"Another check here is called {inner.id}."),
            )
        inner_position = method.inner_position(
            position.inner_position(check.question, check.value, in_list=in_list), inner_location
        )
        yield from _problems(
            inner,
            inner_position,
            (*location, *inner_location),
            written_location,
            scope,
            siblings if method.reads_earlier_results(dict(check.with_)) else earlier,
            locations,
        )
        if inner.id is not None:
            siblings[inner.id] = EARLIER_RESULT


def _cannot_mark_reason(check: Check, position: Position) -> tuple[Location, str] | None:
    """What stops the check marking where it is, at its field at fault; until it can, nothing
    else about it matters."""
    if check.question is not None and check.question not in position.questions:
        return ("question",), f"{check.question} isn't one of the questions."
    check_position = position.for_check(check.question, check.value)
    if check.value is not None and (message := _value_problem(check.value, check_position)):
        return ("value",), message
    if check_position.question is None and not check.method.marks_question_group:
        return ("question",), "Choose the question it checks."
    if message := misfit_reason(check.method, check_position):
        return (), message
    return None


def written_checks(check: Check, location: Location = ()) -> Iterator[tuple[Location, Check]]:
    """``check`` at ``location`` and the checks written in its settings, each where it is."""
    yield location, check
    for inner_location, inner, _ in check.method.inner_checks(check):
        yield from written_checks(inner, (*location, *inner_location))


def _value_problem(value: int, check_position: Position) -> str | None:
    if check_position.value_fixed:
        return "Its place in the list sets the value it checks."
    question = check_position.questions.get(check_position.question or "")
    count = question.value_count if question else 0
    if not count:
        return "Only a question with several values has values to check."
    if value > count:
        return f"The answer has {count} values."
    return None


def misfit_reason(method: Method[Any], position: Position) -> str | None:
    """Why ``method`` can't mark at ``position``, if it can't."""
    if method.can_mark_at(position):
        return None
    return f"{method.label} can't mark {_position_kind(position)}."


def _position_kind(position: Position) -> str:
    question = position.question_model
    if question is None:
        return "a question group"
    if position.value is not None:
        return f"a {question.type} value"
    return f"a {question.label.lower()} question"


def _template_problems(
    check: Check, location: Location, scope: Mapping[str, Shape], earlier: Mapping[str, Shape]
) -> Iterator[tuple[Location, Problem]]:
    templates = check.model_dump(mode="json", by_alias=True, include={"if_", "feedback"})
    scope_shapes = {key: _shape(value) for key, value in check.scope.items()}
    for template_location, error in template_errors(
        templates, {**scope, **scope_shapes, "checks": dict(earlier)}
    ):
        yield (*location, *template_location), Problem(str(error))


def _shape(value: Any) -> Shape:
    if isinstance(value, BaseModel):
        return {key: _shape(item) for key, item in value}
    if isinstance(value, Mapping):
        return {str(key): _shape(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return ListOf(_shape(value[0]) if value else "any")
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    return "text" if isinstance(value, str) else "any"
