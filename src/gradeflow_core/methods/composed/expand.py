"""A composed method's steps with its settings filled in, ready to run."""

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from ...common.problems import invalid_errors
from ..step import Step, load_step
from ..templates import evaluate, is_template, reference, render, top_level_names
from .file import TEMPLATE_KEYS

if TYPE_CHECKING:
    from ..method import Method

EXPANDED_NAMES = frozenset({"settings", "item", "index"})
"""What a method file's references may name before it marks."""


def expand(node: Any, scope: Mapping[str, Any], methods_by_ref: Mapping[str, Method[Any]]) -> Any:
    """``node`` with references to settings and items filled in, and repeated steps repeated."""
    if isinstance(node, Step):
        return node
    if isinstance(node, str):
        expression = reference(node)
        if expression is not None and top_level_names(expression) <= EXPANDED_NAMES:
            return evaluate(expression, scope)
        if expression is None and is_template(node):
            return render(node, scope)
        return node
    if isinstance(node, list):
        return [item for entry in node for item in _expanded_entries(entry, scope, methods_by_ref)]
    if isinstance(node, Mapping):
        if "uses" not in node:
            return {name: expand(value, scope, methods_by_ref) for name, value in node.items()}
        return expand_step(node, scope, methods_by_ref)
    return node


def expand_step(
    node: Mapping[str, Any], scope: Mapping[str, Any], methods_by_ref: Mapping[str, Method[Any]]
) -> Step:
    """``node``, a step as a file writes it, with its references filled in."""
    expanded = {
        name: value if name in TEMPLATE_KEYS else expand(value, scope, methods_by_ref)
        for name, value in node.items()
    }
    with invalid_errors(expanded):
        step = load_step(expanded, methods_by_ref)
    step._scope = dict(scope)  # noqa: SLF001
    return step


def _expanded_entries(
    entry: Any, scope: Mapping[str, Any], methods_by_ref: Mapping[str, Method[Any]]
) -> Iterator[Any]:
    if isinstance(entry, Mapping) and "for_each" in entry:
        repeated_step = {key: value for key, value in entry.items() if key != "for_each"}
        items = list(expand(entry["for_each"], scope, methods_by_ref))
        for index, item in enumerate(items, start=1):
            yield from _expanded_entries(
                repeated_step, {**scope, "item": item, "index": index}, methods_by_ref
            )
        return
    expanded = expand(entry, scope, methods_by_ref)
    if isinstance(expanded, list):
        yield from expanded
    else:
        yield expanded
