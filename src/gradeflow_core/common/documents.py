"""YAML as people write it: loaded with the line of everything in it, and written back."""

import re
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from io import StringIO
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError
from ruamel.yaml.error import YAMLError
from ruamel.yaml.scalarstring import LiteralScalarString

from .exceptions import InvalidError
from .paths import Loc
from .problems import Problem

_DUPLICATE = re.compile(r'found duplicate (?:merge )?key "(.*?)"(?: with value "|$)', re.DOTALL)


class DocumentError(InvalidError):
    """Text that isn't YAML: one problem, saying where."""

    def __init__(self, message: str) -> None:
        super().__init__([Problem(message)])


def _yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 100
    return yaml


def load_document(text: str) -> Any:
    """``text`` as YAML, keeping where each value is. Raises ``DocumentError`` if it isn't YAML."""
    try:
        return _yaml().load(text)
    except YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = f"Line {mark.line + 1}: " if mark is not None else ""
        raise DocumentError(f"{where}{_problem(error)}.") from None


def _problem(error: YAMLError) -> str:
    """What ruamel found wrong, naming a key written twice rather than both its values."""
    problem = getattr(error, "problem", None) or str(error)
    if isinstance(error, DuplicateKeyError) and (found := _DUPLICATE.match(problem)):
        return f'"{found[1]}" is written twice'
    return problem


def plain(node: Any) -> Any:
    """``node`` as plain dicts and lists. Raises ``DocumentError`` for a ``{{ … }}`` left
    unquoted, which YAML reads as a mapping."""
    if isinstance(node, Mapping):
        for key in node:
            if isinstance(key, Mapping):
                line = node.lc.key(key)[0] + 1 if hasattr(node, "lc") else None
                where = f"Line {line}: " if line else ""
                raise DocumentError(f'{where}Quote {{{{ … }}}}, as in "{{{{ inputs.name }}}}".')
        return {str(key): plain(value) for key, value in node.items()}
    if isinstance(node, list):
        return [plain(item) for item in node]
    if isinstance(node, str):
        return str(node)
    return node


def load_yaml(text: str) -> Any:
    """``text`` as YAML, as plain dicts and lists. Raises ``InvalidError`` if it isn't YAML."""
    return plain(load_document(text))


def dump_yaml(data: Any) -> str:
    out = StringIO()
    _yaml().dump(_as_blocks(deepcopy(data)), out)
    return out.getvalue()


def _as_blocks(node: Any) -> Any:
    """``node``, changed in place so that text of several lines is written as a block, as people
    write code."""
    if isinstance(node, str) and "\n" in node:
        return LiteralScalarString(node)
    if isinstance(node, MutableMapping):
        for key, value in node.items():
            node[key] = _as_blocks(value)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            node[index] = _as_blocks(value)
    return node


def _line_of(document: Any, path: Loc) -> int | None:
    """The line, from 1, of the deepest value along ``path`` that ``document`` has."""
    node, line = document, None
    for part in path:
        if isinstance(node, Mapping) and part in node:
            line = node.lc.key(part)[0] + 1 if hasattr(node, "lc") else line
            node = node[part]
        elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
            line = node.lc.item(part)[0] + 1 if hasattr(node, "lc") else line
            node = node[part]
    return line


def on_line(document: Any, path: Loc, message: str) -> str:
    """``message`` after the line of ``path`` in ``document``, if it has one."""
    line = _line_of(document, path)
    return f"Line {line}: {message}" if line is not None else message
