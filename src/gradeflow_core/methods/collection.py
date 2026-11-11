"""Collections of methods, and loading the methods files define.

Each method is one version of a method: checks name it by ``Method.ref``, such as
``gradeflow/key-points@1``.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from operator import attrgetter
from types import MappingProxyType
from typing import Any, Self

from pydantic import ValidationError

from ..common.documents import line_of, load_yaml_with_lines
from ..common.error_formatting import error_message
from ..common.exceptions import GradeFlowError
from ..common.problems import InvalidError, Problem
from ..steps.method import Method
from ..steps.names import method_name, method_ref
from ..steps.step import STEP_OWN_PROPERTIES
from .composed.file import WORKFLOW_REF, MethodFile, methods_used_by_file
from .composed.method import ComposedMethod


class Methods(tuple[Method, ...]):
    """Methods in order, each version of a method once, as checks name them by ``ref``. Raises
    ``InvalidError`` for a ref supplied twice, and ``TypeError`` for a method whose settings share
    a name with a step's own properties, which the editor shows beside them."""

    by_ref: Mapping[str, Method]

    def __new__(cls, methods: Iterable[Method] = ()) -> Self:
        self = super().__new__(cls, methods)
        by_ref: dict[str, Method] = {}
        for method in self:
            if shared_names := STEP_OWN_PROPERTIES.keys() & method.settings.model_fields.keys():
                raise TypeError(
                    f"{type(method).__name__}'s settings share a step's properties: "
                    f"{sorted(shared_names)}"
                )
            if method.ref in by_ref:
                raise InvalidError([Problem(f"Duplicate method {method.ref}.")])
            by_ref[method.ref] = method
        self.by_ref = MappingProxyType(by_ref)
        return self

    def find(self, ref: str) -> Method:
        """The method ``ref`` names. Raises ``GradeFlowError`` if none is it."""
        method = self.by_ref.get(ref)
        if method is None:
            raise GradeFlowError(f"There's no method {ref}.")
        return method

    def latest(self) -> Methods:
        """The highest version of each method, in the order each name first appears."""
        latest_by_name: dict[str, Method] = {}
        for method in self:
            latest_method = latest_by_name.setdefault(method.name, method)
            latest_by_name[method.name] = max(latest_method, method, key=attrgetter("version"))
        return Methods(latest_by_name.values())

    def with_files(self, files: Sequence[tuple[str, str]]) -> Methods:
        """These methods followed by those the ``(name, YAML)`` files define, in order.

        Each name is a namespace and a name, as `method_name` makes them. Files may use these
        methods and each other. Raises ``MethodFileError`` for the first file that doesn't load.
        """
        loaded_files = load_method_files(files, self)
        raise_first_failure(loaded_files.failed)
        return loaded_files.methods

    def dependencies(self, ref: str) -> set[str]:
        """Every ref the method ``ref`` depends on, rejecting missing references and composition
        cycles."""
        graph: dict[str, set[str]] = {}
        pending = [ref]
        while pending:
            current_ref = pending.pop()
            if current_ref in graph:
                continue
            method = self.by_ref.get(current_ref)
            if method is None:
                raise InvalidError([Problem(f"There's no method {current_ref}.")])
            used_refs = (
                methods_used_by_file(method.file) | {WORKFLOW_REF}
                if isinstance(method, ComposedMethod)
                else set()
            )
            graph[current_ref] = used_refs
            pending.extend(used_refs - graph.keys())
        raise_first_failure(_dependency_order(graph)[1])
        return set(graph) - {ref}


class MethodFileError(InvalidError):
    """A method file that doesn't define a method: ``ref`` names the file at fault, by name if its
    version can't be read."""

    def __init__(self, ref: str, problems: list[Problem]) -> None:
        super().__init__(problems)
        self.ref = ref

    def __str__(self) -> str:
        return f"{self.ref}: {super().__str__()}"


@dataclass(frozen=True)
class LoadedFiles:
    """The methods of the method files that load, and why each other file doesn't."""

    methods: Methods
    failed: dict[str, MethodFileError]
    """Why each file that doesn't load fails, by ref, or by name if its version can't be read: a
    problem of its own or of a file it uses."""


def load_method_files(files: Sequence[tuple[str, str]], methods: Methods) -> LoadedFiles:
    """The methods of the ``(name, YAML)`` files that load, after ``methods``, and why each other
    file doesn't. Each name is a namespace and a name, as `method_name` makes them. A file that
    doesn't load stops only the files that use it."""
    well_named_files, failed = [], {}
    for name, yaml_text in files:
        if problems := _name_problems(name):
            failed[name] = MethodFileError(name, problems)
        else:
            well_named_files.append((name, yaml_text))
    loaded_files = load_files(well_named_files, methods)
    return LoadedFiles(loaded_files.methods, failed | loaded_files.failed)


def load_files(files: Sequence[tuple[str, str]], methods: Methods) -> LoadedFiles:
    """`load_method_files`, whatever the names are, as for the built-in methods' own files."""
    methods_by_ref = dict(methods.by_ref)
    parsed_files: dict[str, tuple[str, Any, MethodFile]] = {}
    failed: dict[str, MethodFileError] = {}
    for name, yaml_text in files:
        try:
            document, data = load_yaml_with_lines(yaml_text)
        except InvalidError as error:
            failed[name] = MethodFileError(name, error.problems)
            continue
        version = data.get("version") if isinstance(data, Mapping) else None
        ref = method_ref(name, version) if type(version) is int else name
        if ref in methods_by_ref or ref in parsed_files or ref in failed:
            parsed_files.pop(ref, None)
            failed[ref] = MethodFileError(ref, [Problem("Duplicate method.")])
            continue
        try:
            parsed_files[ref] = name, document, _validated(document, data)
        except InvalidError as error:
            failed[ref] = MethodFileError(ref, error.problems)
    dependencies = {ref: methods_used_by_file(file) for ref, (_, _, file) in parsed_files.items()}
    order, cycle_failures = _dependency_order(
        {ref: used_refs & parsed_files.keys() for ref, used_refs in dependencies.items()}
    )
    failed |= cycle_failures
    for ref in order:
        used_refs = sorted(dependencies[ref])
        if failed_dependency := next(
            (failed[used_ref] for used_ref in used_refs if used_ref in failed), None
        ):
            failed[ref] = failed_dependency
        else:
            try:
                methods_by_ref[ref] = _composed_method(*parsed_files[ref], methods_by_ref)
            except InvalidError as error:
                failed[ref] = MethodFileError(ref, error.problems)
    loaded_methods = (methods_by_ref[ref] for ref in parsed_files if ref not in failed)
    return LoadedFiles(Methods([*methods, *loaded_methods]), failed)


def _name_problems(name: str) -> list[Problem]:
    namespace, separator, short_name = name.partition("/")
    if not separator:
        return [Problem("Name it as a namespace and name, such as intro-biology/term.")]
    try:
        method_name(namespace, short_name)
    except InvalidError as error:
        return error.problems
    return []


def raise_first_failure(failed: Mapping[str, MethodFileError]) -> None:
    """Raises the error of the first file that fails on its own."""
    for failed_ref, failure in failed.items():
        if failure.ref == failed_ref:
            raise failure


def method_of(
    name: str, data: Mapping[str, Any], methods_by_ref: Mapping[str, Method]
) -> ComposedMethod:
    """The method a method file's ``data`` defines, using the methods of ``methods_by_ref``. Raises
    ``InvalidError`` with its problems, which have no line, as data has none."""
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


def _validated(document: Any, data: Any) -> MethodFile:
    try:
        return MethodFile.model_validate(data)
    except ValidationError as error:
        first_error = error.errors()[0]
        path = tuple(first_error["loc"])
        message = error_message(first_error)
        problem = Problem(
            f"{'.'.join(map(str, path))}: {message}" if path else message,
            line=line_of(document, path),
        )
        raise InvalidError([problem]) from None


def _composed_method(
    name: str,
    document: Any,
    file: MethodFile,
    methods_by_ref: Mapping[str, Method],
) -> ComposedMethod:
    """The method ``file`` defines. Raises ``InvalidError`` with each problem on its line of
    ``document``."""
    try:
        return ComposedMethod(name, file, methods_by_ref)
    except InvalidError as error:
        # Its field is a place in the file, not in a step: the line says where it is instead.
        problems = [
            Problem(problem.message, line=line_of(document, problem.field))
            for problem in error.problems
        ]
        raise InvalidError(problems) from None


def _dependency_order(
    dependencies: Mapping[str, set[str]],
) -> tuple[list[str], dict[str, MethodFileError]]:
    """The refs in dependency order, and why those in a cycle fail."""
    cycle_failures: dict[str, MethodFileError] = {}
    while True:
        try:
            return list(TopologicalSorter(dependencies).static_order()), cycle_failures
        except CycleError as error:
            cycle = error.args[1][:-1]
            for index, ref in enumerate(cycle):
                path = [*cycle[index:], *cycle[: index + 1]]
                cycle_problem = Problem(f"Can't use itself: {' → '.join(path)}.")
                cycle_failures[ref] = MethodFileError(ref, [cycle_problem])
            dependencies = {
                ref: used_refs - set(cycle)
                for ref, used_refs in dependencies.items()
                if ref not in cycle
            }
