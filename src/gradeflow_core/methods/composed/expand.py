"""A composed method's steps with its settings filled in, ready to run."""

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from ...common.problems import invalid_errors
from ..step import Step, load_step
from ..templates import evaluate, is_template, reference_expression, render, top_level_names
from .file import TEMPLATE_KEYS

if TYPE_CHECKING:
    from ..method import Method

EXPANDED_NAMES = frozenset({"settings", "item", "index"})
"""What a method file's references may name before it marks."""


def expanded(
    yaml_value: Any, scope: Mapping[str, Any], methods_by_ref: Mapping[str, Method[Any]]
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
    step_data: Mapping[str, Any],
    scope: Mapping[str, Any],
    methods_by_ref: Mapping[str, Method[Any]],
) -> Step:
    """``step_data``, a step as a file writes it, with its references filled in."""
    expanded_step_data = {
        name: value if name in TEMPLATE_KEYS else expanded(value, scope, methods_by_ref)
        for name, value in step_data.items()
    }
    with invalid_errors(expanded_step_data):
        step = load_step(expanded_step_data, methods_by_ref)
    step._scope = dict(scope)  # noqa: SLF001
    return step


def _expanded_entries(
    entry: Any, scope: Mapping[str, Any], methods_by_ref: Mapping[str, Method[Any]]
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
