"""JSON Pointers into a marking as YAML holds it, such as ``/with/steps/0``, the locations they
stand for, and JSON Patch edits to it."""

from collections.abc import Iterable
from typing import Any

import jsonpatch
from jsonpointer import JsonPointer, resolve_pointer

from .schemas import JsonDict

type Location = tuple[str | int, ...]


def json_pointer(location: Location) -> str:
    """The JSON Pointer to ``location``."""
    path: str = JsonPointer.from_parts(location).path
    return path


def location_of(path: str) -> Location:
    """Where the JSON Pointer ``path`` points, list indices as numbers."""
    parts: list[str] = JsonPointer(path).parts
    return tuple(int(part) if part.isdecimal() else part for part in parts)


def is_within(location: Location, ancestor: Location) -> bool:
    """Whether ``location`` is ``ancestor`` or inside it."""
    return location[: len(ancestor)] == ancestor


def value_at(data: Any, path: str) -> Any:
    """The value at the JSON Pointer ``path``; None if there's nothing there."""
    return resolve_pointer(data, path, None)


def patched(data: JsonDict, operations: Iterable[JsonDict]) -> JsonDict:
    """A copy of ``data`` with the JSON Patch ``operations`` applied."""
    result: JsonDict = jsonpatch.apply_patch(data, list(operations))
    return result
