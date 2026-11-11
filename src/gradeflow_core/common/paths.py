"""JSON Pointers into data as YAML holds it, such as ``/with/steps/0``, and JSON Patch edits to
it."""

from collections.abc import Iterable
from typing import Any

import jsonpatch
from jsonpointer import JsonPointer, resolve_pointer, unescape

from .schemas import JsonDict


def pointer(*parts: str | int) -> str:
    """The JSON Pointer to ``parts``, such as ``/with/steps/0``."""
    path: str = JsonPointer.from_parts(parts).path
    return path


def parts_of(path: str) -> list[str]:
    """The parts of the JSON Pointer ``path``, unescaped."""
    parts: list[str] = JsonPointer(path).parts
    return parts


def is_within(path: str, ancestor: str) -> bool:
    """Whether ``path`` is ``ancestor`` or inside it."""
    return path == ancestor or path.startswith(f"{ancestor}/")


def parent(path: str) -> str:
    """The path holding ``path``."""
    return path.rpartition("/")[0]


def last_part(path: str) -> str:
    """The last part of ``path``, unescaped, such as ``0`` of ``/with/steps/0``."""
    part: str = unescape(path.rpartition("/")[2])
    return part


def is_list_item(path: str) -> bool:
    """Whether ``path`` ends with a list index."""
    return last_part(path).isdecimal()


def value_at(data: Any, path: str) -> Any:
    """The value at the JSON Pointer ``path``; None if there's nothing there."""
    return resolve_pointer(data, path, None)


def patched(data: JsonDict, operations: Iterable[JsonDict]) -> JsonDict:
    """A copy of ``data`` with the JSON Patch ``operations`` applied."""
    result: JsonDict = jsonpatch.apply_patch(data, list(operations))
    return result
