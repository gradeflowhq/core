"""How methods are named: ``namespace/name``, and a version of one ``namespace/name@3``."""

import re

from ..common.exceptions import MethodFileError

BUILTIN_NAMESPACE = "gradeflow"
"""The namespace of the methods the core ships, which no one else can have."""

_NAME_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_MAX_LENGTHS = {"namespace": 60, "name": 100}
_REF = re.compile(rf"({_NAME_PATTERN})/({_NAME_PATTERN})@([1-9][0-9]*)")


def method_name(namespace: str, name: str) -> str:
    """``namespace/name``. Raises `MethodFileError` unless both are lowercase words joined by
    hyphens and the namespace isn't the core's."""
    for part_label, part in (("namespace", namespace), ("name", name)):
        if not re.fullmatch(_NAME_PATTERN, part):
            raise MethodFileError(
                f"Use lowercase letters, numbers and hyphens in the {part_label}, "
                "such as intro-biology."
            )
        if len(part) > _MAX_LENGTHS[part_label]:
            raise MethodFileError(
                f"Keep the {part_label} to {_MAX_LENGTHS[part_label]} characters."
            )
    if namespace == BUILTIN_NAMESPACE:
        raise MethodFileError(f"{namespace} is GradeFlow's own.")
    return f"{namespace}/{name}"


def method_ref(name: str, version: int) -> str:
    """How a check names a version of the method `name`: ``intro-biology/term@3``."""
    return f"{name}@{version}"


def split_ref(ref: str) -> tuple[str, str, int]:
    """The namespace, name and version of a method reference. Raises `MethodFileError` if it isn't
    one."""
    match = _REF.fullmatch(ref)
    if match is None:
        raise MethodFileError(f"{ref} isn't a method and version, such as gradeflow/key-points@1.")
    return match[1], match[2], int(match[3])
