"""What stops a marking from marking, each where it is."""

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..common.exceptions import InvalidError
from ..common.paths import Loc, join
from ..common.problems import Problem, placed
from ..questions.question_set import QuestionSet
from .position import Position
from .templates import CHECK_RESULT, ListOf, Shape, TemplateError, template_errors

if TYPE_CHECKING:
    from .check import Check
    from .method import Method


def located_problems(
    marking: Check,
    questions: QuestionSet,
    at: Loc = (),
    *,
    scope: Mapping[str, Shape] = {},
    is_check: Callable[[Loc], bool] | None = None,
) -> list[Problem]:
    """The problems of a marking found at ``at``, as in a mark scheme. Its templates may name
    ``scope`` too, as the checks of a method name its inputs. ``is_check`` says where its data has
    a check, when that isn't where the marking has one."""
    position = Position(questions, marking.question, in_group=marking.question is None)
    try:
        locations = {id(check): loc for loc, check in written_checks(marking, at)}
        found = list(_problems(marking, position, at, at, scope, {}, locations))
        if marking.question is not None:
            outside = (marking.questions_read() | marking.questions_marked()) - {marking.question}
            if outside:
                read = ", ".join(sorted(outside))
                found.append((at, Problem(f"Use a group marking to read or award {read}.")))
    except InvalidError as error:
        # A check generated inside a composed method: name its settings, not where they are written.
        message = " ".join(map(str, error.problems))
        return [Problem(f"{marking.method.label}: {message}", join(*at))]
    except TemplateError as error:
        return [Problem(f"{marking.method.label}: {error}", join(*at))]
    editable = set(locations.values())
    is_check = is_check or (lambda loc: loc in editable)
    return [placed(problem, loc, is_check) for loc, problem in found]


def _problems(
    check: Check,
    position: Position,
    at: Loc,
    owner: Loc,
    scope: Mapping[str, Shape],
    earlier: Mapping[str, Shape],
    locations: Mapping[int, Loc],
) -> Iterator[tuple[Loc, Problem]]:
    """Each problem with where it is, as YAML holds the marking. ``owner`` is where the check
    written in the marking is: the check itself, or the one using the composed method that
    generated it, which has its problems as a whole."""
    if id(check) in locations:
        at = owner = locations[id(check)]

    def here(loc: Loc, problem: Problem) -> tuple[Loc, Problem]:
        return (loc if at == owner else owner), problem

    if cant := _cant_mark_here(check, position):
        field, message = cant
        yield here((*at, *field), Problem(message))
        return
    method = check.method
    own = position.at(check.question, check.value)
    yield from (here(*found) for found in _template_problems(check, at, scope, earlier))
    for found in method.problems(check, own):
        yield here((*at, "with", *found.field), found)
    lists: dict[Loc, dict[str, Any]] = {}
    for loc, child, in_list in method.validation_children(check):
        slot = loc[:-1] if in_list else loc
        siblings = lists.setdefault(slot, dict(earlier))
        if child.id is not None and child.id in siblings and in_list:
            yield here((*at, *loc, "id"), Problem(f"Another check here is called {child.id}."))
        inside = method.inside(position.inside(check.question, check.value, in_list=in_list), loc)
        yield from _problems(
            child,
            inside,
            (*at, *loc),
            owner,
            scope,
            siblings if method.ordered(dict(check.with_)) else earlier,
            locations,
        )
        if child.id is not None:
            siblings[child.id] = CHECK_RESULT


def _cant_mark_here(check: Check, position: Position) -> tuple[Loc, str] | None:
    """What stops the check marking where it is, at its field at fault; until it can, nothing
    else about it matters."""
    if check.question is not None and check.question not in position.questions:
        return ("question",), f"{check.question} isn't in the supplied questions."
    own = position.at(check.question, check.value)
    if check.value is not None and (message := _value_problem(check.value, own)):
        return ("value",), message
    if own.question is None and not check.method.group:
        return ("question",), "Choose the question it checks."
    if message := misfit(check.method, own):
        return (), message
    return None


def written_checks(check: Check, at: Loc = ()) -> Iterator[tuple[Loc, Check]]:
    """``check`` at ``at`` and the checks written in its settings, each where it is."""
    yield at, check
    for loc, child, _ in check.method.nested(check):
        yield from written_checks(child, (*at, *loc))


def _value_problem(value: int, own: Position) -> str | None:
    if own.value_fixed:
        return "Its place in the list sets the value it checks."
    question = own.questions.get(own.question or "")
    count = question.value_count if question else 0
    if not count:
        return "Only a question with several values has values to check."
    if value > count:
        return f"The answer has {count} values."
    return None


def misfit(method: Method[Any], position: Position) -> str | None:
    """Why ``method`` can't mark at ``position``, if it can't."""
    if method.fits(position):
        return None
    if (
        position.answer_type in method.question_types
        and position.question is not None
        and position.value is None
    ):
        if position.answer_shape == "multiple":
            return (
                f"{method.label} marks one answer, and this question reads each answer as "
                "several. Put it inside Best Answer."
            )
        return f"{method.label} marks an answer read as several, which this question doesn't."
    return f"{method.label} can't mark {_what(position)}."


def _what(position: Position) -> str:
    question = position.question_model
    if question is None:
        return "a question group"
    if position.value is not None:
        return f"a {question.type} value"
    return f"a {question.label.lower()} question"


def _template_problems(
    check: Check, at: Loc, scope: Mapping[str, Shape], earlier: Mapping[str, Shape]
) -> Iterator[tuple[Loc, Problem]]:
    written = check.model_dump(mode="json", by_alias=True, include={"if_", "feedback"})
    own = {key: _shape(value) for key, value in check.scope.items()}
    for loc, error in template_errors(written, {**scope, **own, "checks": dict(earlier)}):
        yield (*at, *loc), Problem(str(error))


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
