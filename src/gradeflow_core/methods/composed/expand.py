"""A composed method's steps with its settings filled in, ready to run."""

from collections.abc import Iterator, Mapping
from dataclasses import replace
from functools import cached_property
from typing import Any, ClassVar

from pydantic import JsonValue, PrivateAttr

from ...common.paths import pointer, value_at
from ...common.problems import InvalidError, Problem
from ...steps.method import Method
from ...steps.settings import Settings
from ...steps.step import METHODS_CONTEXT_KEY, Step, valid_step
from ...steps.templates import evaluate, is_template, reference_expression, render, top_level_names
from .file import TEMPLATE_KEYS

EXPANDED_NAMES = frozenset({"settings", "item", "index"})
"""What a method file's references may name before it marks."""


class ComposedSettings(Settings):
    """A composed method's settings, which fill its file's steps in using the methods their step
    was loaded with."""

    _workflow: ClassVar[Mapping[str, JsonValue]]
    """The workflow its method's file writes, as a step."""
    _methods_by_ref: Mapping[str, Method] = PrivateAttr()

    def __init_subclass__(cls, *, workflow: Mapping[str, JsonValue], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._workflow = workflow

    def model_post_init(self, context: Any) -> None:
        self._methods_by_ref = context[METHODS_CONTEXT_KEY]

    @cached_property
    def expansion(self) -> Step:
        """The workflow a step with these settings is. Raises ``InvalidError`` or
        ``TemplateError`` if they make none."""
        return expanded_step(self._workflow, file_scope(self), self._methods_by_ref)


def file_scope(settings: ComposedSettings) -> dict[str, Any]:
    """What a file's templates see of a step: its settings."""
    # The model's own iteration includes the cached expansion.
    return {"settings": {name: getattr(settings, name) for name in type(settings).model_fields}}


def expanded(
    yaml_value: JsonValue,
    scope: Mapping[str, Any],
    methods_by_ref: Mapping[str, Method],
) -> Any:
    """``yaml_value`` with references to settings and items filled in, and repeated steps
    repeated."""
    if isinstance(yaml_value, str):
        expression = reference_expression(yaml_value)
        if expression is not None and top_level_names(expression) <= EXPANDED_NAMES:
            return evaluate(expression, scope)
        if expression is None and is_template(yaml_value):
            return render(yaml_value, scope)
        return yaml_value
    if isinstance(yaml_value, list):
        return [
            item for entry in yaml_value for item in _expanded_entries(entry, scope, methods_by_ref)
        ]
    if isinstance(yaml_value, Mapping):
        if "uses" not in yaml_value:
            return {
                name: expanded(value, scope, methods_by_ref) for name, value in yaml_value.items()
            }
        return expanded_step(yaml_value, scope, methods_by_ref)
    return yaml_value


def expanded_step(
    step_data: Mapping[str, JsonValue],
    scope: Mapping[str, Any],
    methods_by_ref: Mapping[str, Method],
) -> Step:
    """``step_data``, a step as a file writes it, with its references filled in. Raises
    ``InvalidError`` if what fills them makes no step, each problem at the setting at fault."""
    expanded_step_data = {
        name: value if name in TEMPLATE_KEYS else expanded(value, scope, methods_by_ref)
        for name, value in step_data.items()
    }
    try:
        return valid_step(expanded_step_data, methods_by_ref, scope)
    except InvalidError as error:
        raise InvalidError(
            [_at_setting(problem, step_data) for problem in error.problems]
        ) from None


def _at_setting(problem: Problem, step_data: Mapping[str, JsonValue]) -> Problem:
    """``problem`` of the step ``step_data`` writes, at the setting that fills its field; else of
    the settings as a whole."""
    expression = reference_expression(value_at(step_data, problem.where)) or ""
    root, *names = expression.split(".")
    field = pointer(names[0]) if root == "settings" and names else ""
    return replace(problem, step_path="", field=field)


def _expanded_entries(
    entry: JsonValue,
    scope: Mapping[str, Any],
    methods_by_ref: Mapping[str, Method],
) -> Iterator[Any]:
    if isinstance(entry, Mapping) and "for_each" in entry:
        repeated_step = {key: value for key, value in entry.items() if key != "for_each"}
        items = list(expanded(entry["for_each"], scope, methods_by_ref))
        for index, item in enumerate(items, start=1):
            yield from _expanded_entries(
                repeated_step, {**scope, "item": item, "index": index}, methods_by_ref
            )
        return
    expanded_entry = expanded(entry, scope, methods_by_ref)
    if isinstance(expanded_entry, list):
        yield from expanded_entry
    else:
        yield expanded_entry
