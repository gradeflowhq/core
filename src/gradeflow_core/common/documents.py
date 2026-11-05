"""YAML as people write it: loaded with the line of everything in it, and written back."""

import re
from collections.abc import Mapping
from io import StringIO
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.composer import Composer, ComposerError, MaxDepthExceededError
from ruamel.yaml.constructor import DuplicateKeyError
from ruamel.yaml.error import YAMLError
from ruamel.yaml.nodes import ScalarNode
from ruamel.yaml.representer import RoundTripRepresenter

from .exceptions import InvalidError
from .paths import Location
from .problems import Problem

_MAX_DEPTH = 100
"""How deeply YAML may nest; the built-in methods nest 12 deep."""
_DUPLICATE_KEY = re.compile(r'found duplicate (?:merge )?key "(.*?)"(?: with value "|$)', re.DOTALL)


class _Composer(Composer):
    def return_alias(self, a: Any) -> Any:
        mark = self.parser.last_event.start_mark
        raise ComposerError(None, None, "Anchors and aliases aren't supported", mark)


class _Representer(RoundTripRepresenter):
    def ignore_aliases(self, data: Any) -> bool:
        """Never writes aliases, as loading refuses them."""
        return True

    def represent_str(self, data: str) -> ScalarNode:
        """``data``, as a block if it has several lines, as people write code."""
        style = "|" if "\n" in data else None
        return self.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Representer.add_representer(str, _Representer.represent_str)


def _new_yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.Composer = _Composer
    yaml.Representer = _Representer
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 100
    yaml.max_depth = _MAX_DEPTH
    return yaml


def load_yaml_with_lines(text: str) -> Any:
    """``text`` as YAML, keeping where each value is. Raises ``InvalidError`` if it isn't YAML."""
    try:
        return _new_yaml().load(text)
    except YAMLError as error:
        position = getattr(error, "problem_mark", None)
        where = f"Line {position.line + 1}: " if position is not None else ""
        raise InvalidError([Problem(f"{where}{_yaml_problem(error)}.")]) from None


def _yaml_problem(error: YAMLError) -> str:
    """What ruamel found wrong, naming a key written twice rather than both its values."""
    if isinstance(error, MaxDepthExceededError):
        return f"It nests more than {_MAX_DEPTH} levels deep"
    problem = getattr(error, "problem", None) or str(error)
    if isinstance(error, DuplicateKeyError) and (match := _DUPLICATE_KEY.match(problem)):
        return f'"{match[1]}" is written twice'
    return problem


def as_plain_data(node: Any) -> Any:
    """``node`` as plain dicts and lists. Raises ``InvalidError`` for a ``{{ … }}`` left
    unquoted, which YAML reads as a mapping."""
    if isinstance(node, Mapping):
        for key in node:
            if isinstance(key, Mapping):
                line = node.lc.key(key)[0] + 1 if hasattr(node, "lc") else None
                where = f"Line {line}: " if line else ""
                raise InvalidError(
                    [Problem(f'{where}Quote {{{{ … }}}}, as in "{{{{ inputs.name }}}}".')]
                )
        return {str(key): as_plain_data(value) for key, value in node.items()}
    if isinstance(node, list):
        return [as_plain_data(item) for item in node]
    if isinstance(node, str):
        return str(node)
    return node


def load_yaml(text: str) -> Any:
    """``text`` as YAML, as plain dicts and lists. Raises ``InvalidError`` if it isn't YAML."""
    return as_plain_data(load_yaml_with_lines(text))


def dump_yaml(data: Any) -> str:
    buffer = StringIO()
    _new_yaml().dump(data, buffer)
    return buffer.getvalue()


def _line_of(document: Any, path: Location) -> int | None:
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


def with_line(document: Any, path: Location, message: str) -> str:
    """``message`` after the line of ``path`` in ``document``, if it has one."""
    line = _line_of(document, path)
    return f"Line {line}: {message}" if line is not None else message
