"""YAML as people write it: loaded with the line of everything in it, and written back."""

import re
from collections.abc import Mapping
from io import StringIO
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.composer import Composer, ComposerError, MaxDepthExceededError
from ruamel.yaml.constructor import DuplicateKeyError
from ruamel.yaml.error import YAMLError
from ruamel.yaml.nodes import ScalarNode
from ruamel.yaml.representer import RoundTripRepresenter

from .paths import parts_of
from .problems import InvalidError, Problem

_MAX_DEPTH = 100
"""How deeply YAML may nest; the built-in methods nest 12 deep."""
_DUPLICATE_KEY = re.compile(r'found duplicate (?:merge )?key "(.*?)"(?: with value "|$)', re.DOTALL)


class _Composer(Composer):
    def return_alias(self, a: Any) -> Any:
        position = self.parser.last_event.start_mark
        raise ComposerError(None, None, "Anchors and aliases aren't supported", position)


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


def load_yaml_with_lines(text: str) -> tuple[Any, Any]:
    """``text`` as YAML: the document, which keeps where each value is, and its plain dicts and
    lists. Raises ``InvalidError`` if it isn't YAML."""
    try:
        document = _new_yaml().load(text)
    except YAMLError as error:
        position = getattr(error, "problem_mark", None)
        line = position.line + 1 if position is not None else None
        raise InvalidError([Problem(f"{_yaml_error_message(error)}.", line=line)]) from None
    return document, _as_plain_data(document)


def _yaml_error_message(error: YAMLError) -> str:
    """What ruamel found wrong, naming a key written twice rather than both its values."""
    if isinstance(error, MaxDepthExceededError):
        return f"It nests more than {_MAX_DEPTH} levels deep"
    message = getattr(error, "problem", None) or str(error)
    if isinstance(error, DuplicateKeyError) and (match := _DUPLICATE_KEY.match(message)):
        return f'"{match[1]}" is written twice'
    return message


def _as_plain_data(value: Any) -> Any:
    """``value`` as plain dicts and lists. Raises ``InvalidError`` for a ``{{ … }}`` left
    unquoted, which YAML reads as a mapping."""
    if isinstance(value, CommentedMap):
        for key in value:
            if isinstance(key, Mapping):
                message = 'Quote {{ … }}, as in "{{ settings.name }}".'
                raise InvalidError([Problem(message, line=value.lc.key(key)[0] + 1)])
        return {str(key): _as_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_plain_data(item) for item in value]
    if isinstance(value, str):
        return str(value)
    return value


def load_yaml(text: str) -> Any:
    """``text`` as YAML, as plain dicts and lists. Raises ``InvalidError`` if it isn't YAML."""
    return load_yaml_with_lines(text)[1]


def dump_yaml(data: Any) -> str:
    buffer = StringIO()
    _new_yaml().dump(data, buffer)
    return buffer.getvalue()


def line_of(document: Any, path: str) -> int | None:
    """The line, from 1, of the deepest value along the JSON Pointer ``path`` that ``document``
    has."""
    value, line = document, None
    for part in parts_of(path):
        if isinstance(value, Mapping) and part in value:
            line = value.lc.key(part)[0] + 1 if hasattr(value, "lc") else line
            value = value[part]
        elif isinstance(value, list) and part.isdecimal() and int(part) < len(value):
            line = value.lc.item(int(part))[0] + 1 if hasattr(value, "lc") else line
            value = value[int(part)]
    return line
