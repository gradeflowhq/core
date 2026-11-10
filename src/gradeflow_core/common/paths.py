"""Paths into a marking as YAML holds it, such as ``with.checks.0``: keys and list indices."""

from collections.abc import Iterable, Mapping
from typing import Any

from .schemas import JsonDict

type Location = tuple[str | int, ...]


def path_parts(path: str) -> list[str | int]:
    return [int(part) if part.isdecimal() else part for part in path.split(".") if part]


def join_path(*parts: object) -> str:
    return ".".join(str(part) for part in parts if part != "")


def get_at(value: Any, path: str) -> Any:
    """The value at ``path``; None if there's nothing there."""
    return get_by_parts(value, path_parts(path))


def get_by_parts(value: Any, parts: Iterable[str | int]) -> Any:
    """The value at the keys and indices ``parts``; None if there's nothing there."""
    for part in parts:
        if isinstance(value, list):
            exists = isinstance(part, int) and part < len(value)
        else:
            exists = isinstance(value, Mapping) and part in value
        if not exists:
            return None
        value = value[part]
    return value


def set_at(data: JsonDict, path: str, value: Any) -> None:
    """Sets the value at ``path``, making the objects on the way that aren't there; at the root,
    replaces the data's contents."""
    if not path:
        data.clear()
        data.update(value)
        return
    *parent_parts, last_part = path_parts(path)
    container: Any = data
    for part in parent_parts:
        if isinstance(container, dict) and not isinstance(container.get(part), dict | list):
            container[part] = {}
        container = container[part]
    container[last_part] = value
