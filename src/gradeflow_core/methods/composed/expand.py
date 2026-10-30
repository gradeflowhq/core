"""A composed method's checks with its inputs filled in, ready to mark."""

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from ...common.problems import as_invalid
from ..check import Check, check_of
from ..templates import evaluate, is_template, reference, render, roots
from .file import TEMPLATE_KEYS

if TYPE_CHECKING:
    from ..method import Method

EXPANDED = frozenset({"inputs", "each", "index"})
"""What a method file's references may name before it marks."""


def expand(node: Any, scope: Mapping[str, Any], lookup: Mapping[str, Method[Any]]) -> Any:
    """The body with references to inputs and items filled in, repeated checks repeated, and
    retaining conditions for the common check evaluator. Each check carries its template scope."""
    if isinstance(node, Check):
        return node
    if isinstance(node, str):
        expression = reference(node)
        if expression is not None and roots(expression) <= EXPANDED:
            return evaluate(expression, scope)
        if expression is None and is_template(node):
            return render(node, scope)
        return node
    if isinstance(node, list):
        return [item for entry in node for item in _entries(entry, scope, lookup)]
    if isinstance(node, Mapping):
        if "uses" not in node:
            return {name: expand(value, scope, lookup) for name, value in node.items()}
        expanded = {
            name: value if name in TEMPLATE_KEYS else expand(value, scope, lookup)
            for name, value in node.items()
        }
        with as_invalid(expanded):
            return check_of(expanded, lookup).scoped(scope)
    return node


def _entries(
    entry: Any, scope: Mapping[str, Any], lookup: Mapping[str, Method[Any]]
) -> Iterator[Any]:
    if isinstance(entry, Mapping) and "for_each" in entry:
        rest = {key: value for key, value in entry.items() if key != "for_each"}
        items = list(expand(entry["for_each"], scope, lookup))
        for index, each in enumerate(items, start=1):
            yield from _entries(rest, {**scope, "each": each, "index": index}, lookup)
        return
    expanded = expand(entry, scope, lookup)
    if isinstance(expanded, list):
        yield from expanded
    else:
        yield expanded
