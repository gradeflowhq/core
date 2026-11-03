"""Collections of methods, and loading the methods files define.

Each method is one version of a method: checks name it by ``Method.ref``, such as
``gradeflow/key-points@1``.
"""

from collections.abc import Iterable, Mapping, Sequence
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Any, Self

from pydantic import ValidationError

from ..common.documents import YamlError, as_plain_data, load_yaml_with_lines, with_line
from ..common.error_formatting import error_message
from ..common.exceptions import GradeFlowError, MethodFileError
from ..common.paths import join
from .composed.file import MethodFile, MethodFileFieldError, methods_used, methods_used_by_file
from .composed.method import ComposedMethod
from .method import Method
from .names import method_name, method_ref


class Methods(tuple[Method[Any], ...]):
    """Methods in order, each version of a method once, as checks name them by ``ref``. Raises
    ``MethodFileError`` for a ref supplied twice."""

    _by_ref: Mapping[str, Method[Any]]

    def __new__(cls, methods: Iterable[Method[Any]] = ()) -> Self:
        self = super().__new__(cls, methods)
        by_ref: dict[str, Method[Any]] = {}
        for method in self:
            if method.ref in by_ref:
                raise MethodFileError(f"Duplicate method {method.ref}.")
            by_ref[method.ref] = method
        self._by_ref = MappingProxyType(by_ref)
        return self

    @property
    def by_ref(self) -> Mapping[str, Method[Any]]:
        """The methods by ref."""
        return self._by_ref

    def find(self, ref: str) -> Method[Any]:
        """The method ``ref`` names. Raises ``GradeFlowError`` if none is it."""
        method = self.by_ref.get(ref)
        if method is None:
            raise GradeFlowError(f"There's no method {ref}.")
        return method

    def latest(self) -> Methods:
        """The highest version of each method, in the order each name first appears."""
        latest_by_name: dict[str, Method[Any]] = {}
        for method in self:
            if (
                method.name not in latest_by_name
                or method.version > latest_by_name[method.name].version
            ):
                latest_by_name[method.name] = method
        return Methods(latest_by_name.values())

    def with_files(self, files: Sequence[tuple[str, str]]) -> Methods:
        """These methods followed by those the ``(name, YAML)`` files define, in order.

        Each name is a namespace and a name, as `method_name` makes them. Files may use these
        methods and each other.
        """
        for name, _ in files:
            namespace, separator, short_name = name.partition("/")
            if not separator:
                raise MethodFileError(
                    f"{name}: Name it as a namespace and name, such as intro-biology/term."
                )
            try:
                method_name(namespace, short_name)
            except MethodFileError as error:
                raise MethodFileError(f"{name}: {error}") from None
        return load_files(files, self)

    def dependencies(self, ref: str) -> set[str]:
        """Every ref the method ``ref`` depends on, rejecting missing references and composition
        cycles."""
        graph: dict[str, set[str]] = {}
        pending = [ref]
        while pending:
            current = pending.pop()
            if current in graph:
                continue
            method = self.by_ref.get(current)
            if method is None:
                raise MethodFileError(f"There's no method {current}.")
            references = (
                methods_used(method.file.combination()) | methods_used_by_file(method.file)
                if isinstance(method, ComposedMethod)
                else set()
            )
            graph[current] = references
            pending.extend(references - graph.keys())
        _dependency_order(graph)
        return set(graph) - {ref}


def load_files(files: Sequence[tuple[str, str]], methods: Methods) -> Methods:
    """`Methods.with_files`, whatever the names are, as for the built-in methods' own files."""
    methods_by_ref = dict(methods.by_ref)
    parsed = {}
    for name, text in files:
        try:
            document, file = _parsed_file(text)
        except MethodFileError as error:
            raise MethodFileError(f"{name}: {error}") from None
        ref = method_ref(name, file.version)
        if ref in methods_by_ref or ref in parsed:
            raise MethodFileError(f"Duplicate method {ref}.")
        parsed[ref] = name, document, file
    dependencies = {ref: methods_used_by_file(file) for ref, (_, _, file) in parsed.items()}
    for ref, references in dependencies.items():
        for missing in sorted(references - methods_by_ref.keys() - parsed.keys()):
            raise MethodFileError(f"{ref}: There's no method {missing}.")
    for ref in _dependency_order({ref: used & parsed.keys() for ref, used in dependencies.items()}):
        try:
            methods_by_ref[ref] = _composed_method(*parsed[ref], methods_by_ref)
        except MethodFileError as error:
            raise MethodFileError(f"{ref}: {error}") from None
    return Methods([*methods, *(methods_by_ref[ref] for ref in parsed)])


def method_of(
    name: str, data: Mapping[str, Any], methods_by_ref: Mapping[str, Method[Any]]
) -> Method[Any]:
    """The method a method file's ``data`` defines, using the methods of ``methods_by_ref``. Raises
    ``MethodFileError`` with its problem and no line, as data has none."""
    return _composed_method(name, data, _validated(data, data), methods_by_ref)


def rewrite_references(data: Any, replacements: Mapping[str, str]) -> Any:
    """Copy a method file or a marking, as YAML holds it, replacing the named method references."""
    if isinstance(data, Mapping):
        return {
            key: replacements.get(value, value)
            if key == "uses" and isinstance(value, str)
            else rewrite_references(value, replacements)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [rewrite_references(item, replacements) for item in data]
    return data


def _parsed_file(text: str) -> tuple[Any, MethodFile]:
    try:
        document = load_yaml_with_lines(text)
        data = as_plain_data(document)
    except YamlError as error:
        raise MethodFileError(str(error)) from None
    return document, _validated(document, data)


def _validated(document: Any, data: Any) -> MethodFile:
    try:
        return MethodFile.model_validate(data)
    except ValidationError as error:
        first_error = error.errors()[0]
        path = tuple(first_error["loc"])
        raise MethodFileError(
            with_line(document, path, f"{join(*path)}: {error_message(first_error)}")
        ) from None


def _composed_method(
    name: str, document: Any, file: MethodFile, methods_by_ref: Mapping[str, Method[Any]]
) -> ComposedMethod:
    try:
        return ComposedMethod(name, file, methods_by_ref)
    except MethodFileFieldError as error:
        raise MethodFileError(with_line(document, error.path, str(error))) from None


def _dependency_order(dependencies: Mapping[str, set[str]]) -> list[str]:
    try:
        return list(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        cycle = error.args[1]
        raise MethodFileError(f"{cycle[0]} can't use itself: {' → '.join(cycle)}.") from None
