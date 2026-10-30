"""Collections of methods, and loading the methods files define.

Each method is one version of a method: checks name it by ``Method.ref``, such as
``gradeflow/key-points@1``.
"""

from collections.abc import Iterable, Mapping, Sequence
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Any, overload

from pydantic import ValidationError

from ..common.documents import DocumentError, load_document, on_line, plain
from ..common.error_formatting import error_message
from ..common.exceptions import GradeFlowError, MethodFileError
from .composed.file import Fault, MethodFile, methods_used
from .composed.method import ComposedMethod
from .method import Method
from .names import method_name, method_ref


class Methods(Sequence[Method[Any]]):
    """Methods in order, each version of a method once, as checks name them by ``ref``. Raises
    ``MethodFileError`` for a ref supplied twice."""

    def __init__(self, methods: Iterable[Method[Any]] = ()) -> None:
        self._methods = tuple(methods)
        found: dict[str, Method[Any]] = {}
        for method in self._methods:
            if method.ref in found:
                raise MethodFileError(f"Duplicate method {method.ref}.")
            found[method.ref] = method
        self._by_ref: Mapping[str, Method[Any]] = MappingProxyType(found)

    @property
    def by_ref(self) -> Mapping[str, Method[Any]]:
        """The methods by ref."""
        return self._by_ref

    @overload
    def __getitem__(self, index: int) -> Method[Any]: ...
    @overload
    def __getitem__(self, index: slice) -> Methods: ...
    def __getitem__(self, index: int | slice) -> Method[Any] | Methods:
        return Methods(self._methods[index]) if isinstance(index, slice) else self._methods[index]

    def __len__(self) -> int:
        return len(self._methods)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Methods):
            return NotImplemented
        return self._methods == other._methods

    def __hash__(self) -> int:
        return hash(self._methods)

    def __repr__(self) -> str:
        return f"<Methods ({len(self)})>"

    def find(self, ref: str) -> Method[Any]:
        """The method ``ref`` names. Raises ``GradeFlowError`` if none is it."""
        method = self.by_ref.get(ref)
        if method is None:
            raise GradeFlowError(f"There's no method {ref}.")
        return method

    def latest(self) -> Methods:
        """The highest version of each method, in the order each name first appears."""
        found: dict[str, Method[Any]] = {}
        for method in self:
            if method.name not in found or method.version > found[method.name].version:
                found[method.name] = method
        return Methods(found.values())

    def with_files(self, files: Sequence[tuple[str, str]]) -> Methods:
        """These methods followed by those the ``(name, YAML)`` files define, in order.

        Each name is a namespace and a name, as `method_name` makes them. Files may use these
        methods and each other.
        """
        for name, _ in files:
            namespace, slash, rest = name.partition("/")
            if not slash:
                raise MethodFileError(
                    f"{name}: Name it as a namespace and name, such as intro-biology/term."
                )
            try:
                method_name(namespace, rest)
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
                methods_used(method.file.body()) | _definition_references(method.file)
                if isinstance(method, ComposedMethod)
                else set()
            )
            graph[current] = references
            pending.extend(references - graph.keys())
        _ordered(graph)
        return set(graph) - {ref}


def load_files(files: Sequence[tuple[str, str]], methods: Methods) -> Methods:
    """`Methods.with_files`, whatever the names are, as for the built-in methods' own files."""
    lookup = dict(methods.by_ref)
    parsed = {}
    for name, text in files:
        try:
            document, file = _file(text)
        except MethodFileError as error:
            raise MethodFileError(f"{name}: {error}") from None
        ref = method_ref(name, file.version)
        if ref in lookup or ref in parsed:
            raise MethodFileError(f"Duplicate method {ref}.")
        parsed[ref] = name, document, file
    dependencies = {ref: _definition_references(file) for ref, (_, _, file) in parsed.items()}
    for ref, references in dependencies.items():
        for missing in sorted(references - lookup.keys() - parsed.keys()):
            raise MethodFileError(f"{ref}: There's no method {missing}.")
    for ref in _ordered({ref: used & parsed.keys() for ref, used in dependencies.items()}):
        try:
            lookup[ref] = _composed(*parsed[ref], lookup)
        except MethodFileError as error:
            raise MethodFileError(f"{ref}: {error}") from None
    return Methods([*methods, *(lookup[ref] for ref in parsed)])


def method_of(name: str, data: Mapping[str, Any], lookup: Mapping[str, Method[Any]]) -> Method[Any]:
    """The method a method file's ``data`` defines, using the methods of ``lookup``. Raises
    ``MethodFileError`` with its problem, said without a line, as data has none."""
    return _composed(name, data, _validated(data, data), lookup)


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


def _file(text: str) -> tuple[Any, MethodFile]:
    try:
        document = load_document(text)
        data = plain(document)
    except DocumentError as error:
        raise MethodFileError(str(error)) from None
    return document, _validated(document, data)


def _validated(document: Any, data: Any) -> MethodFile:
    try:
        return MethodFile.model_validate(data)
    except ValidationError as error:
        first = error.errors()[0]
        path = tuple(first["loc"])
        where = ".".join(map(str, path))
        raise MethodFileError(on_line(document, path, f"{where}: {error_message(first)}")) from None


def _composed(
    name: str, document: Any, file: MethodFile, lookup: Mapping[str, Method[Any]]
) -> ComposedMethod:
    try:
        return ComposedMethod(name, file, lookup)
    except Fault as fault:
        raise MethodFileError(on_line(document, fault.path, str(fault))) from None


def _definition_references(file: MethodFile) -> set[str]:
    """Composition references, including defaults; examples are evaluated separately."""
    return methods_used(file.model_dump(exclude={"examples"}))


def _ordered(dependencies: Mapping[str, set[str]]) -> list[str]:
    try:
        return list(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        cycle = error.args[1]
        raise MethodFileError(f"{cycle[0]} can't use itself: {' → '.join(cycle)}.") from None
