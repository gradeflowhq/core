"""Paths into a marking as YAML holds it, such as ``with.checks.0``: keys and list indices."""

from collections.abc import Mapping
from typing import Any

from .schemas import JsonDict

type Loc = tuple[str | int, ...]


def parts(path: str) -> list[str | int]:
    return [int(part) if part.isdecimal() else part for part in path.split(".") if part]


def join(*segments: object) -> str:
    return ".".join(str(segment) for segment in segments if segment != "")


def get(node: Any, path: str) -> Any:
    """The value at ``path``; None if there's nothing there."""
    for part in parts(path):
        if isinstance(node, list):
            found = isinstance(part, int) and part < len(node)
        else:
            found = isinstance(node, Mapping) and part in node
        if not found:
            return None
        node = node[part]
    return node


def put(data: JsonDict, path: str, value: Any) -> None:
    """Sets the value at ``path``, making the objects on the way that aren't there; at the root,
    replaces the data's contents."""
    if not path:
        data.clear()
        data.update(value)
        return
    *parent, last = parts(path)
    node: Any = data
    for part in parent:
        if isinstance(node, dict) and not isinstance(node.get(part), dict | list):
            node[part] = {}
        node = node[part]
    node[last] = value
