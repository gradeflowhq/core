"""A marking being edited one check at a time, as YAML holds it, before it is a whole check."""

import re
from collections.abc import Container, Iterator, Mapping
from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
from itertools import chain, count
from typing import Any, TypeGuard

from pydantic import ConfigDict, ValidationError, create_model
from pydantic_core import PydanticUndefined

from ..common.documents import as_plain_data, load_yaml_with_lines, with_line
from ..common.exceptions import EditingError, InvalidError
from ..common.paths import Location, get, join, path_parts, set_at
from ..common.problems import Problem, validation_problems
from ..common.schemas import JsonDict, json_schema
from ..common.text import format_fraction
from ..methods.catalogue import BUILTINS
from ..methods.check import CHECK_FIELDS, CHECK_KEYS, Check, Marking, check_of, how_many_checks
from ..methods.collection import Methods
from ..methods.method import Method
from ..methods.position import Position
from ..methods.problems import located_problems, misfit_reason, written_checks
from ..methods.settings import literal
from ..methods.templates import Shape, names_read, unknown_check_message
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from .conditions import Condition, condition_template, read_condition

POSITION_KEYS = ("id", "question", "value", "name", "weight", "if", "read_only")
"""What a check has because of where it is, which stays when its method changes."""
EMPTY = (None, "")
"""What a field left empty holds."""
_SCHEMA_POSITION_FIELDS = tuple(
    field.name for field in fields(Position) if field.name != "questions"
)
"""What of its position a check's schema depends on: all but the questions, a draft's own."""


def compatible_methods(position: Position, *, methods: Methods = BUILTINS) -> Methods:
    """The methods a check at ``position`` can use, in their order."""
    return Methods(method for method in methods if _fits(method, position))


@dataclass(frozen=True)
class ChecksSetting:
    """A setting of a check that holds checks: one at ``path``, or a list of them there if
    ``is_list``."""

    name: str
    path: str
    title: str
    is_list: bool
    minimum: int
    maximum: int | None
    checks: tuple[str, ...]
    """The paths of the checks in it."""

    @property
    def new_check_path(self) -> str:
        """The path a check added to it gets."""
        return join(self.path, len(self.checks)) if self.is_list else self.path

    @property
    def is_full(self) -> bool:
        if not self.is_list:
            return bool(self.checks)
        return self.maximum is not None and len(self.checks) >= self.maximum


@dataclass(frozen=True)
class _WrittenCheck:
    """A check of a marking as written, and where it is."""

    path: str
    method: Method[Any] | None
    position: Position
    enclosing_position: Position
    """Where it is before its own question and value: what the checks holding it chose."""
    siblings: tuple[str, ...]
    """The paths of the checks before it in its list, when they run in order."""


class MarkingDraft:
    """A marking of ``question``, or of a question group without one, as it is being edited. Its
    checks are at paths such as ``with.checks.0``; ``""`` is the marking itself. A check's fields
    are its method's settings and the keys a check has there, as one object. A change to a check
    or setting that isn't there raises ``EditingError``."""

    def __init__(
        self,
        data: Mapping[str, Any],
        questions: QuestionSet,
        question: QuestionId | None,
        *,
        methods: Methods = BUILTINS,
    ) -> None:
        self.questions, self.question = questions, question
        self._methods, self._methods_by_reference = methods, methods.by_ref
        self._data: JsonDict = deepcopy(dict(data))
        self._schemas: dict[tuple[object, ...], JsonDict] = {}
        self._cached_validation: tuple[Check | None, list[Problem]] | None = None
        self._source: tuple[Any, JsonDict] | None = None
        """The YAML it was read from, and the marking as read, which places its problems on their
        lines until it changes."""

    @classmethod
    def from_yaml(
        cls,
        text: str,
        questions: QuestionSet,
        question: QuestionId | None,
        *,
        methods: Methods = BUILTINS,
    ) -> MarkingDraft:
        """The marking ``text`` writes of ``question``, which it needn't name. Raises
        ``InvalidError`` for text that isn't one check, or names another question."""
        document = load_yaml_with_lines(text)
        data = as_plain_data(document)
        if not isinstance(data, dict):
            raise InvalidError(
                [Problem("Write one check, starting with uses: and its method's name.")]
            )
        names_other_question = "question" in data and data["question"] != question
        data = (
            {**data, "question": question}
            if question
            else {key: value for key, value in data.items() if key != "question"}
        )
        draft = cls(data, questions, question, methods=methods)
        draft._source = (document, draft.data)
        if names_other_question:
            fix_message = (
                f"Set question to {question}."
                if question
                else "Remove question; a question group's checks name their own."
            )
            question_problem = Problem(with_line(document, ("question",), fix_message))
            raise InvalidError(list(dict.fromkeys([question_problem, *draft.problems])))
        return draft

    @classmethod
    def new(
        cls,
        reference: str,
        questions: QuestionSet,
        question: QuestionId | None,
        *,
        methods: Methods = BUILTINS,
    ) -> MarkingDraft:
        """A draft of new marking with the method ``reference`` names. Raises ``EditingError`` if it
        can't mark it."""
        draft = cls({}, questions, question, methods=methods)
        draft.replace(_new_check(draft._method(reference), draft.position(""), names_question=True))
        return draft

    @property
    def data(self) -> JsonDict:
        """The marking as YAML holds it, as written."""
        return deepcopy(self._data)

    def replace(self, data: Mapping[str, Any]) -> None:
        self._data = deepcopy(dict(data))
        self._cached_validation = None

    def paths(self) -> list[str]:
        """The paths of its checks, each before those it holds."""
        return [written_check.path for written_check in self._walk()]

    def is_check(self, path: str) -> bool:
        return self._written_check(path) is not None

    def method(self, path: str) -> Method[Any] | None:
        node = get(self._data, path)
        return (
            self._methods_by_reference.get(str(node.get("uses")))
            if isinstance(node, Mapping)
            else None
        )

    def position(self, path: str, question: QuestionId | None = None) -> Position:
        """Where the check at ``path`` is, or would be; reading ``question`` if given. Raises
        ``EditingError`` for a path no check can be at."""
        written_check = self._written_check(path)
        position = written_check.position if written_check else self._new_position(path)
        return position.for_check(question, None) if question else position

    def _new_position(self, path: str) -> Position:
        """Where a check added at ``path`` would be."""
        if not path:
            return self._root
        list_path = _list_path_of(path)
        owner_check = self._written_check(_owner_of(list_path))
        setting_path, in_list = path_parts(list_path)[-2:], list_path != path
        fields = (
            owner_check.method.settings.model_fields if owner_check and owner_check.method else {}
        )
        holds = (
            how_many_checks(fields[setting_path[1]])
            if len(setting_path) == 2 and setting_path[0] == "with" and setting_path[1] in fields
            else None
        )
        if owner_check is None or holds is None or (holds == "many") != in_list:
            raise _not_a_check(path)
        assert owner_check.method is not None  # it has the setting
        return _position_inside(
            owner_check.method,
            owner_check.position,
            tuple(path_parts(path)[len(path_parts(owner_check.path)) :]),
        )

    @property
    def _root(self) -> Position:
        return Position(self.questions, self.question, in_group=self.question is None)

    def _walk(self) -> Iterator[_WrittenCheck]:
        """Each check of the marking, each before those it holds."""

        def walk(
            node: Mapping[str, Any],
            path: str,
            enclosing_position: Position,
            siblings: tuple[str, ...],
        ) -> Iterator[_WrittenCheck]:
            method = self._methods_by_reference.get(str(node["uses"]))
            position = enclosing_position.for_check(*self._question_and_value(node))
            yield _WrittenCheck(path, method, position, enclosing_position, siblings)
            settings = node.get("with")
            if method is None or not isinstance(settings, Mapping):
                return
            runs_in_order = method.reads_earlier_results(settings)
            for name, field in method.settings.model_fields.items():
                holds, value, location = how_many_checks(field), settings.get(name), ("with", name)
                if holds == "one" and _is_check(value):
                    inner_position = _position_inside(method, position, location)
                    yield from walk(value, join(path, *location), inner_position, ())
                elif holds == "many" and isinstance(value, list):
                    earlier_paths: tuple[str, ...] = ()
                    for number, check in enumerate(value):
                        if not _is_check(check):
                            continue
                        item_path, item_location = (
                            join(path, *location, number),
                            (*location, number),
                        )
                        inner_position = _position_inside(method, position, item_location)
                        yield from walk(check, item_path, inner_position, earlier_paths)
                        earlier_paths = (*earlier_paths, item_path) if runs_in_order else ()

        return walk(self._data, "", self._root, ()) if _is_check(self._data) else iter(())

    def _written_check(self, path: str) -> _WrittenCheck | None:
        return next(
            (written_check for written_check in self._walk() if written_check.path == path), None
        )

    def _question_and_value(self, node: Mapping[str, Any]) -> tuple[Any, Any]:
        """The question and value a check names for itself."""
        return node.get("question"), node.get("value")

    def methods(self, path: str, question: QuestionId | None = None) -> Methods:
        """The methods a check at ``path`` can use."""
        return compatible_methods(self.position(path, question), methods=self._methods)

    def schema(self, path: str) -> JsonDict:
        """The JSON schema of the fields of the check at ``path``; a condition among them once
        there is something for one to read, or the check has one."""
        method, position = str(self._check_data(path)["uses"]), self.position(path)
        written_check = self._written_check(path)
        enclosing_position = written_check.enclosing_position if written_check else position
        key = (
            method,
            *(getattr(position, name) for name in _SCHEMA_POSITION_FIELDS),
            enclosing_position.question,
            enclosing_position.value,
        )
        if key not in self._schemas:
            self._schemas[key] = _check_schema(self._method(method), position, enclosing_position)
        schema = self._schemas[key]
        if "if" in self._check_data(path) or self._condition_can_read(path):
            return schema
        return {
            **schema,
            "properties": {
                key: value for key, value in schema["properties"].items() if key != "if"
            },
        }

    def values(self, path: str) -> JsonDict:
        """The fields of the check at ``path``, read as its place reads them; its condition as a
        ``Condition`` if made of rows."""
        node, method = deepcopy(self._check_data(path)), self.method(path)
        if method:
            node["with"] = method.settings_read_at(node.get("with") or {}, self.position(path))
        values = fields_of(node)
        if isinstance(values.get("if"), str):
            ids = {earlier: self._check_id(earlier) for earlier in self.earlier_checks(path)}
            paths = {id_: earlier for earlier, id_ in ids.items() if id_}
            if (condition := read_condition(values["if"])) is not None and all(
                row.check in paths for row in condition.rows
            ):
                values["if"] = replace(
                    condition,
                    rows=tuple(replace(row, check=paths[row.check]) for row in condition.rows),
                )
        return values

    def update(self, path: str, values: Mapping[str, Any]) -> None:
        """Sets fields of the check at ``path``; those not given stay. An optional setting left
        empty is unset; a required one stays empty, for its problem to say so. A key a check has
        is left out when empty or at its default there. A ``Condition`` names the checks it reads
        by their paths, which get IDs."""
        node = self._check_data(path)
        merged = {**self.values(path), **values}
        if isinstance(condition := merged.get("if"), Condition):
            merged["if"] = self._condition_template(path, condition)
        schema = self.schema(path)
        defaults = {
            key: property_schema.get("default")
            for key, property_schema in schema["properties"].items()
        }
        unset_keys = {
            key
            for key, value in merged.items()
            if value in EMPTY
            and key not in schema.get("required", ())
            and defaults.get(key) is None
        }
        check = {
            "uses": node["uses"],
            "with": {
                key: value for key, value in merged.items() if key not in (*CHECK_KEYS, *unset_keys)
            },
            **{
                key: value
                for key, value in merged.items()
                if key in CHECK_FIELDS and value not in (*EMPTY, defaults.get(key))
            },
        }
        self._put(path, {key: check[key] for key in CHECK_KEYS if key in check})

    def change_method(self, path: str, reference: str) -> None:
        """Gives the check at ``path`` another method, keeping where it is. When the new method
        holds checks, as Combination and Not do, the old check becomes its first, if it fits."""
        old_check = deepcopy(self._check_data(path))
        new_check = _new_check(
            self._method(reference), self.position(path), names_question=not path
        )
        self._put(
            path, {key: old_check[key] for key in POSITION_KEYS if key in old_check} | new_check
        )
        old_inner_check = {
            key: value for key, value in old_check.items() if key not in POSITION_KEYS
        }
        for setting in self.checks_settings(path):
            if not setting.checks and old_inner_check["uses"] in {
                method.ref for method in self.methods(setting.new_check_path)
            }:
                self._put(setting.path, [old_inner_check] if setting.is_list else old_inner_check)
                return

    def add(self, path: str, reference: str, question: QuestionId | None = None) -> None:
        """Adds a check of the method ``reference`` names at ``path``, a setting's
        ``new_check_path``, reading ``question`` if given."""
        settings = self.checks_settings(_owner_of(_list_path_of(path)))
        setting = next((setting for setting in settings if setting.new_check_path == path), None)
        if setting is None or setting.is_full:
            raise EditingError(f"{path} isn't a place for another check.")
        new_check = _new_check(
            self._method(reference),
            self.position(path, question),
            names_question=question is not None,
        )
        self._put(
            setting.path,
            [*(get(self._data, setting.path) or []), new_check] if setting.is_list else new_check,
        )

    def remove(self, path: str) -> str:
        """Removes the check at ``path``, inside the marking; the path of the check that held
        it."""
        if not path or not self.is_check(path):
            raise EditingError(f"{path} isn't a check inside this marking.")
        parent, last = _parent_of(path)
        get(self._data, parent).pop(last)
        self._cached_validation = None
        return _owner_of(_list_path_of(path))

    def move(self, path: str, offset: int) -> None:
        """Moves the check at ``path`` within its list; at either end, no further."""
        parent, index = _parent_of(path) if path else ("", None)
        siblings = get(self._data, parent)
        if not self.is_check(path) or not isinstance(index, int) or not isinstance(siblings, list):
            raise EditingError(f"{path} isn't a check in a list.")
        if 0 <= (other_index := index + offset) < len(siblings):
            siblings[index], siblings[other_index] = siblings[other_index], siblings[index]
            self._cached_validation = None

    def _put(self, path: str, value: Any) -> None:
        set_at(self._data, path, value)
        self._cached_validation = None

    def checks_settings(self, path: str) -> list[ChecksSetting]:
        """The settings of the check at ``path`` that hold checks."""
        method = self.method(path)
        if method is None:
            return []
        properties = self.schema(path)["properties"]
        settings = []
        for name, field in method.settings_model_at(self.position(path)).model_fields.items():
            holds = how_many_checks(field)
            if holds is None:
                continue
            setting_path = join(path, "with", name)
            property_schema = properties.get(name, {})
            value = get(self._data, setting_path)
            if holds == "many":
                checks = tuple(join(setting_path, index) for index in range(len(value or [])))
            else:
                checks = (setting_path,) if value else ()
            settings.append(
                ChecksSetting(
                    name=name,
                    path=setting_path,
                    title=field.title or name,
                    is_list=holds == "many",
                    minimum=property_schema.get("minItems", 0),
                    maximum=property_schema.get("maxItems"),
                    checks=checks,
                )
            )
        return settings

    def label(self, path: str) -> str:
        """The check's name, the value of the answer it reads, or else its method's label."""
        node, position = self._check_data(path), self.position(path)
        if name := self._resolved_value(node.get("name")):
            return str(name)
        value = (
            position.value
            if position.value_fixed
            else path and self._resolved_value(node.get("value"))
        )
        if value:
            return f"Value {value}"
        method = self.method(path)
        return method.label if method else str(node["uses"])

    def notes(self, path: str) -> list[str]:
        """What the check's place says about it, beside its label: the question it reads, that
        it gives no marks or else its weight, and when it counts."""
        values = self.values(path)
        notes = [str(values["question"])] if values.get("question") else []
        weight = values.get("weight", Check.model_fields["weight"].default)
        if values.get("read_only"):
            notes.append("Read only")
        elif weight != 1:
            notes.append(
                f"Worth {format_fraction(weight)}"
                if weight > 0
                else f"Penalty {format_fraction(-weight)}"
            )
        if condition := values.get("if"):
            notes.append(
                condition.describe(self.label)
                if isinstance(condition, Condition)
                else "Only when its condition holds"
            )
        return notes

    def describe(self, path: str) -> str:
        """What the check at ``path`` does, as soon as it is a check itself, even while a check
        elsewhere in the marking isn't finished."""
        check = self._checks().get(path)
        if check is None and self.is_check(path):
            try:
                check = check_of(self._check_data(path), self._methods_by_reference)
            except ValidationError:
                return ""
        return check.describe() if check else ""

    def earlier_checks(self, path: str) -> list[str]:
        """Paths of checks whose results the condition at ``path`` may read: those before it
        in its list, and in the lists holding it."""
        checks = self._checks()
        marking, problems = self._validation()
        if marking is not None and not problems and (target := checks.get(path)) is not None:
            preceding = _checks_run_before(marking, target, []) or []
            paths = {id(check): check_path for check_path, check in checks.items()}
            return [
                check_path
                for check in preceding
                if (check_path := paths.get(id(check))) is not None
            ]
        return self._earlier_checks_as_written(path)

    def _earlier_checks_as_written(self, path: str) -> list[str]:
        """``earlier_checks`` as the data is written, for a marking that isn't a check yet."""
        return [
            sibling
            for written_check in self._walk()
            if _is_within(path, written_check.path)
            for sibling in written_check.siblings
        ]

    def finalized(self) -> Marking:
        """The marking, once nothing is wrong with it. Raises ``EditingError`` with its first
        problem."""
        marking, problems = self._validation()
        if problems:
            raise EditingError(str(problems[0]))
        assert marking is not None  # with no problems, it is a check
        return marking

    @property
    def problems(self) -> list[Problem]:
        """What is wrong with the marking, each at the check and field at fault; as read from
        YAML, each after its line instead."""
        problems = self._validation()[1]
        if self._source is None or self._source[1] != self._data:
            return problems
        document = self._source[0]
        on_lines = (
            Problem(with_line(document, _yaml_location(problem, self._data), str(problem)))
            for problem in problems
        )
        return list(dict.fromkeys(on_lines))

    @property
    def _template_names(self) -> Mapping[str, Shape]:
        """What its templates may name beside earlier checks."""
        return {}

    def _resolved_value(self, value: Any) -> Any:
        """``value`` as the marking reads it."""
        return value

    def _condition_can_read(self, path: str) -> bool:
        """Whether a condition of the check at ``path`` has something to read."""
        return bool(self.earlier_checks(path)) or any(self._template_names.values())

    def _validation(self) -> tuple[Check | None, list[Problem]]:
        if self._cached_validation is None:
            self._cached_validation = self._validation_of(self._data)
        return self._cached_validation

    def _validation_of(self, data: JsonDict) -> tuple[Check | None, list[Problem]]:
        try:
            check = check_of(data, self._methods_by_reference)
        except ValidationError as error:
            position_problems = self._position_problems()
            problems = validation_problems(error, data, self._is_check_at)
            return None, [
                *position_problems,
                *(
                    problem
                    for problem in problems
                    if not any(
                        _is_within(problem.check, misfit_problem.check)
                        for misfit_problem in position_problems
                    )
                ),
                *map(self._condition_order_problem, self._unknown_check_problems()),
            ]
        return check, [
            self._condition_order_problem(problem)
            for problem in located_problems(
                check, self.questions, scope=self._template_names, is_check=self._is_check_at
            )
        ]

    def _unknown_check_problems(self) -> Iterator[Problem]:
        """The conditions naming a check not before them, found before the marking is whole."""
        for written_check in self._walk():
            condition = get(self._data, written_check.path).get("if")
            if not isinstance(condition, str):
                continue
            earlier_ids = {
                self._check_id(earlier)
                for earlier in self._earlier_checks_as_written(written_check.path)
            }
            for name in sorted(names_read(condition, "checks") - earlier_ids):
                yield Problem(unknown_check_message(name), written_check.path, ("if",))

    def _position_problems(self) -> list[Problem]:
        """The checks whose method can't mark where they are, found before the marking is whole."""
        return [
            Problem(message, written_check.path)
            for written_check in self._walk()
            if written_check.method
            and (message := misfit_reason(written_check.method, written_check.position))
        ]

    def _condition_order_problem(self, problem: Problem) -> Problem:
        """A problem with a condition that reads a check after it, as what to do about it."""
        condition = get(self._data, join(problem.check, "if")) if problem.field == ("if",) else None
        if not isinstance(condition, str):
            return problem
        later_paths_by_id = {
            id_: written_check.path
            for written_check in self._walk()
            if any(_is_within(problem.check, sibling) for sibling in written_check.siblings)
            and (id_ := self._check_id(written_check.path))
        }
        later_checks_read = [
            later_paths_by_id[name]
            for name in sorted(names_read(condition, "checks"))
            if name in later_paths_by_id
        ]
        if not later_checks_read:
            return problem
        label = self.label(later_checks_read[0])
        return replace(problem, message=f"{label} must come before this check.")

    def _is_check_at(self, location: Location) -> bool:
        return self.is_check(join(*location))

    def _checks(self) -> dict[str, Check]:
        """The checks of the whole marking by path, once it is a check."""
        marking = self._validation()[0]
        return (
            {join(*location): check for location, check in written_checks(marking)}
            if marking
            else {}
        )

    def _check_data(self, path: str) -> JsonDict:
        if not self.is_check(path):
            raise _not_a_check(path)
        node: JsonDict = get(self._data, path)
        return node

    def _method(self, reference: str) -> Method[Any]:
        method = self._methods_by_reference.get(reference)
        if method is None:
            raise EditingError(f"There's no method {reference}.")
        return method

    def _check_id(self, path: str) -> str | None:
        node = get(self._data, path)
        return str(node["id"]) if isinstance(node, Mapping) and node.get("id") else None

    def _condition_template(self, path: str, condition: Condition) -> str:
        """The condition as the check's ``if``, giving each check it reads an ID if it has none."""
        readable = set(self.earlier_checks(path))
        rows = tuple(
            replace(row, check=self._check_id(row.check) or self._give_id(row.check))
            for row in condition.rows
            if row.check in readable
        )
        return condition_template(replace(condition, rows=rows))

    def _give_id(self, path: str) -> str:
        """Gives the check at ``path`` an ID made from its label."""
        taken = {id_ for check_path in self.paths() if (id_ := self._check_id(check_path))}
        stem = re.sub(r"\W+", "_", self.label(path).lower()).strip("_") or "check"
        stem = stem if stem[0].isalpha() else f"check_{stem}"
        name = first_unused_name(stem, taken)
        node = {**self._check_data(path), "id": name}
        set_at(self._data, path, {key: node[key] for key in CHECK_KEYS if key in node})
        return name


def _yaml_location(problem: Problem, data: JsonDict) -> Location:
    """Where the problem is in the YAML: a field the check doesn't have as a key is a setting,
    under its ``with``, and a problem with no field is at the check's ``uses``."""
    node, field = get(data, problem.check), problem.field
    if not field:
        return (*path_parts(problem.check), "uses")
    is_check_key = field[0] in CHECK_KEYS or (isinstance(node, Mapping) and field[0] in node)
    return (*path_parts(problem.check), *(field if is_check_key else ("with", *field)))


def _not_a_check(path: str) -> EditingError:
    return EditingError(f"{path} isn't a check of this marking.")


def fields_of(check: Mapping[str, Any]) -> JsonDict:
    """A check's settings and keys as one, as a ``Problem``'s field names them."""
    return {
        **(check.get("with") or {}),
        **{key: value for key, value in check.items() if key not in ("uses", "with")},
    }


def field_path(check: str, key: str) -> str:
    """Where the field ``key`` of the check at ``check`` is: a key a check has is its own, and
    any other is a setting."""
    return join(check, key) if key in CHECK_FIELDS else join(check, "with", key)


def _is_check(node: Any) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(node, Mapping) and "uses" in node


def _position_inside(method: Method[Any], position: Position, location: Location) -> Position:
    """Where a check at ``location`` of the settings of a check of ``method`` at ``position`` is,
    before its own place."""
    in_list = isinstance(location[-1], int)
    return method.inner_position(position.inner_position(None, None, in_list=in_list), location)


def _checks_run_before(check: Check, target: Check, earlier: list[Check]) -> list[Check] | None:
    if check is target:
        return earlier
    run_by_list: dict[Location, list[Check]] = {}
    for location, inner, in_list in check.method.expanded_checks(check):
        list_location = location[:-1] if in_list else location
        siblings = run_by_list.setdefault(list_location, [])
        earlier_checks = (
            [*earlier, *siblings]
            if check.method.reads_earlier_results(dict(check.with_))
            else earlier
        )
        if (run_before := _checks_run_before(inner, target, earlier_checks)) is not None:
            return run_before
        siblings.append(inner)
    return None


def _new_check(method: Method[Any], position: Position, *, names_question: bool) -> JsonDict:
    """A new check of ``method`` at ``position``, naming its question if ``names_question``. Raises
    ``EditingError`` for a method that can't mark there."""
    if not _fits(method, position):
        raise EditingError(f"{method.label} can't mark here.")
    new_check: JsonDict = {"uses": method.ref, "with": method.initial_settings(position)}
    if names_question and position.question:
        return {"question": position.question, **new_check}
    return new_check


def _fits(method: Method[Any], position: Position) -> bool:
    return method.level != "internal" and method.can_mark_at(position)


def first_unused_name(stem: str, taken: Container[object], separator: str = "_") -> str:
    """``stem``, or else the first of it numbered from 2 after ``separator`` that isn't taken."""
    numbered = (f"{stem}{separator}{number}" for number in count(2))
    return next(name for name in chain([stem], numbered) if name not in taken)


def _parent_of(path: str) -> tuple[str, str | int]:
    *parent, last = path_parts(path)
    return join(*parent), last


def _list_path_of(path: str) -> str:
    """The list a check in one, like ``with.checks.0``, is in: ``with.checks``."""
    segments = path_parts(path)
    return join(*segments[:-1]) if segments and isinstance(segments[-1], int) else path


def _owner_of(setting_path: str) -> str:
    """The check a setting, like ``with.checks``, belongs to: here the marking itself, ``""``."""
    return join(*path_parts(setting_path)[:-2])


def _is_within(path: str, root: str) -> bool:
    return not root or path == root or path.startswith(f"{root}.")


def _check_schema(
    method: Method[Any], position: Position, enclosing_position: Position
) -> JsonDict:
    """The JSON schema of the fields of a check of ``method`` at ``position``: those that place it
    in a question group's list first, then its method's settings there, then the other keys a
    check there may have. A check in a question group's list chooses from the supplied
    questions, unless a check holding it, at ``enclosing_position``, chose its question; so with
    values."""
    trailing_keys = ["if_", "feedback"]
    if position.in_list:
        trailing_keys = ["name", "weight", *trailing_keys]
    # A method of several questions names its questions in its own checks.
    placement_keys = (
        ["question", "read_only"]
        if position.in_group
        and position.in_list
        and not method.marks_question_group
        and enclosing_position.question is None
        else []
    )
    check_fields = Check.model_fields
    fields: dict[str, Any] = {
        key: (check_fields[key].annotation, check_fields[key]) for key in placement_keys
    }
    if "question" in fields and position.questions:
        # A copy without its default: merging field infos would keep the default.
        required_question = copy(check_fields["question"])
        required_question.default = PydanticUndefined
        fields["question"] = (literal(position.questions), required_question)
    settings = method.settings_model_at(position).model_fields
    fields |= {name: (field.annotation, field) for name, field in settings.items()}
    fields |= {key: (check_fields[key].annotation, check_fields[key]) for key in trailing_keys}
    question = position.questions.get(position.question) if position.question else None
    # A method for values questions marks all of an answer's values, so its checks don't pick one.
    marks_all_values = method.question_types == {"values"}
    if (
        question is not None
        and question.value_count
        and position.in_list
        and not position.value_fixed
        and enclosing_position.value is None
        and not marks_all_values
    ):
        fields["value"] = (
            literal(range(1, question.value_count + 1)) | None,
            check_fields["value"],
        )
    return json_schema(
        create_model(
            f"{method.label} check",
            __config__=ConfigDict(validate_by_name=True, title=method.label),
            **fields,
        )
    )
