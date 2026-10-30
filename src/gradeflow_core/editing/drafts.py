"""A marking being edited one check at a time, as YAML holds it, before it is a whole check."""

import re
from collections.abc import Container, Iterator, Mapping
from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
from itertools import chain, count
from typing import Any, TypeGuard

from pydantic import ConfigDict, ValidationError, create_model
from pydantic_core import PydanticUndefined

from ..common.documents import load_document, on_line, plain
from ..common.exceptions import EditingError, InvalidError
from ..common.paths import Loc, get, join, parts, put
from ..common.problems import Problem, validation_problems
from ..common.schemas import JsonDict, json_schema
from ..common.text import format_fraction
from ..methods.catalogue import BUILTINS
from ..methods.check import CHECK_FIELDS, KEYS, Check, Marking, check_of, holds_checks
from ..methods.collection import Methods
from ..methods.method import Method
from ..methods.position import Position
from ..methods.problems import located_problems, misfit, written_checks
from ..methods.settings import literal
from ..methods.templates import Shape, named, unknown_check
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from .conditions import Condition, read_condition, written_condition

PLACE = ("id", "question", "value", "name", "weight", "if", "read_only")
"""What a check has because of where it is, which stays when its method changes."""
EMPTY = (None, "")
"""What a field left empty holds."""
_SCHEMA_KEY = tuple(field.name for field in fields(Position) if field.name != "questions")
"""What of its position a check's schema depends on: all but the questions, a draft's own."""


def compatible_methods(position: Position, *, methods: Methods = BUILTINS) -> Methods:
    """The methods a check at ``position`` can use, in their order."""
    return Methods(
        method for method in methods if method.level != "internal" and method.fits(position)
    )


@dataclass(frozen=True)
class Slot:
    """Where a check holds checks: one at ``path``, or a list of them there if ``many``."""

    name: str
    path: str
    title: str
    many: bool
    minimum: int
    maximum: int | None
    checks: tuple[str, ...]
    """The paths of the checks in it."""

    @property
    def new(self) -> str:
        """The path a check added to it gets."""
        return join(self.path, len(self.checks)) if self.many else self.path

    @property
    def full(self) -> bool:
        if not self.many:
            return bool(self.checks)
        return self.maximum is not None and len(self.checks) >= self.maximum


@dataclass(frozen=True)
class _Found:
    """A check of a marking as written, and where it is."""

    path: str
    method: Method[Any] | None
    position: Position
    holder: Position
    """Where it is before its own question and value: what the checks holding it chose."""
    siblings: tuple[str, ...]
    """The paths of the checks before it in its list, when they run in order."""


class MarkingDraft:
    """A marking of ``question``, or of a question group without one, as it is being edited. Its
    checks are at paths such as ``with.checks.0``; ``""`` is the marking itself. A check's fields
    are its method's settings and the keys a check has there, as one object. A change to a check
    or slot that isn't there raises ``EditingError``."""

    def __init__(
        self,
        data: Mapping[str, Any],
        questions: QuestionSet,
        question: QuestionId | None,
        *,
        methods: Methods = BUILTINS,
    ) -> None:
        self.questions, self.question = questions, question
        self._methods, self._lookup = methods, methods.by_ref
        self._data: JsonDict = deepcopy(dict(data))
        self._schemas: dict[tuple[object, ...], JsonDict] = {}
        self._validated: tuple[Check | None, list[Problem]] | None = None
        self._yaml: tuple[Any, JsonDict] | None = None
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
        document = load_document(text)
        data = plain(document)
        if not isinstance(data, dict):
            raise InvalidError(
                [Problem("Write one check, starting with uses: and its method's name.")]
            )
        names_other = "question" in data and data["question"] != question
        data = (
            {**data, "question": question}
            if question
            else {k: v for k, v in data.items() if k != "question"}
        )
        draft = cls(data, questions, question, methods=methods)
        draft._yaml = (document, draft.data)
        if names_other:
            fix = (
                f"Set question to {question}."
                if question
                else "Remove question; a question group's checks name their own."
            )
            fixed = Problem(on_line(document, ("question",), fix))
            raise InvalidError(list(dict.fromkeys([fixed, *draft.problems])))
        return draft

    @classmethod
    def new(
        cls,
        ref: str,
        questions: QuestionSet,
        question: QuestionId | None,
        *,
        methods: Methods = BUILTINS,
    ) -> MarkingDraft:
        """A draft of new marking with the method ``ref`` names. Raises ``EditingError`` if it
        can't mark it."""
        draft = cls({}, questions, question, methods=methods)
        draft.replace(_new_check(draft._method(ref), draft.position(""), named=True))
        return draft

    @property
    def data(self) -> JsonDict:
        """The marking as YAML holds it, as written."""
        return deepcopy(self._data)

    def replace(self, data: Mapping[str, Any]) -> None:
        self._data = deepcopy(dict(data))
        self._validated = None

    def paths(self) -> list[str]:
        """The paths of its checks, each before those it holds."""
        return [found.path for found in self._walk()]

    def is_check(self, path: str) -> bool:
        return self._found(path) is not None

    def method(self, path: str) -> Method[Any] | None:
        node = get(self._data, path)
        return self._lookup.get(str(node.get("uses"))) if isinstance(node, Mapping) else None

    def position(self, path: str, question: QuestionId | None = None) -> Position:
        """Where the check at ``path`` is, or would be; reading ``question`` if given. Raises
        ``EditingError`` for a path no check can be at."""
        found = self._found(path)
        position = found.position if found else self._new_position(path)
        return position.at(question, None) if question else position

    def _new_position(self, path: str) -> Position:
        """Where a check added at ``path`` would be."""
        if not path:
            return self._root
        slot = _slot_of(path)
        owner, steps, in_list = self._found(_owner_of(slot)), parts(slot)[-2:], slot != path
        fields = owner.method.settings.model_fields if owner and owner.method else {}
        held = (
            holds_checks(fields[steps[1]])
            if len(steps) == 2 and steps[0] == "with" and steps[1] in fields
            else None
        )
        if owner is None or held is None or (held == "many") != in_list:
            raise _not_a_check(path)
        assert owner.method is not None  # it has the setting
        return _inside(owner.method, owner.position, tuple(parts(path)[len(parts(owner.path)) :]))

    @property
    def _root(self) -> Position:
        return Position(self.questions, self.question, in_group=self.question is None)

    def _walk(self) -> Iterator[_Found]:
        """Each check of the marking, each before those it holds."""

        def walk(
            node: Mapping[str, Any], at: str, outer: Position, siblings: tuple[str, ...]
        ) -> Iterator[_Found]:
            method = self._lookup.get(str(node["uses"]))
            position = outer.at(*self._place(node))
            yield _Found(at, method, position, outer, siblings)
            settings = node.get("with")
            if method is None or not isinstance(settings, Mapping):
                return
            ordered = method.ordered(settings)
            for name, info in method.settings.model_fields.items():
                held, value, loc = holds_checks(info), settings.get(name), ("with", name)
                if held == "one" and _is_check(value):
                    yield from walk(value, join(at, *loc), _inside(method, position, loc), ())
                elif held == "many" and isinstance(value, list):
                    before: tuple[str, ...] = ()
                    for number, check in enumerate(value):
                        if not _is_check(check):
                            continue
                        path, inside = join(at, *loc, number), (*loc, number)
                        yield from walk(check, path, _inside(method, position, inside), before)
                        before = (*before, path) if ordered else ()

        return walk(self._data, "", self._root, ()) if _is_check(self._data) else iter(())

    def _found(self, path: str) -> _Found | None:
        return next((found for found in self._walk() if found.path == path), None)

    def _place(self, node: Mapping[str, Any]) -> tuple[Any, Any]:
        """The question and value a check names for itself."""
        return node.get("question"), node.get("value")

    def methods(self, path: str, question: QuestionId | None = None) -> Methods:
        """The methods a check at ``path`` can use."""
        return compatible_methods(self.position(path, question), methods=self._methods)

    def schema(self, path: str) -> JsonDict:
        """The JSON schema of the fields of the check at ``path``; a condition among them once
        there is something for one to read, or the check has one."""
        method, position = str(self._node(path)["uses"]), self.position(path)
        found = self._found(path)
        holder = found.holder if found else position
        key = (
            method,
            *(getattr(position, name) for name in _SCHEMA_KEY),
            holder.question,
            holder.value,
        )
        if key not in self._schemas:
            self._schemas[key] = _check_schema(self._method(method), position, holder)
        schema = self._schemas[key]
        if "if" in self._node(path) or self._readable(path):
            return schema
        return {
            **schema,
            "properties": {k: v for k, v in schema["properties"].items() if k != "if"},
        }

    def values(self, path: str) -> JsonDict:
        """The fields of the check at ``path``, read as its place reads them; its condition as a
        ``Condition`` if made of rows."""
        node, method = deepcopy(self._node(path)), self.method(path)
        if method:
            node["with"] = method.read_at(node.get("with") or {}, self.position(path))
        values = fields_of(node)
        if isinstance(values.get("if"), str):
            ids = {at: self._id(at) for at in self.earlier(path)}
            paths = {id_: at for at, id_ in ids.items() if id_}
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
        node = self._node(path)
        merged = {**self.values(path), **values}
        if isinstance(condition := merged.get("if"), Condition):
            merged["if"] = self._condition(path, condition)
        schema = self.schema(path)
        defaults = {key: prop.get("default") for key, prop in schema["properties"].items()}
        unset = {
            key
            for key, value in merged.items()
            if value in EMPTY
            and key not in schema.get("required", ())
            and defaults.get(key) is None
        }
        check = {
            "uses": node["uses"],
            "with": {key: value for key, value in merged.items() if key not in (*KEYS, *unset)},
            **{
                key: value
                for key, value in merged.items()
                if key in CHECK_FIELDS and value not in (*EMPTY, defaults.get(key))
            },
        }
        put(self._data, path, {key: check[key] for key in KEYS if key in check})
        self._validated = None

    def retype(self, path: str, ref: str) -> None:
        """Gives the check at ``path`` another method, keeping where it is. When the new method
        holds checks, as Combination and Not do, the old check becomes its first, if it fits."""
        old = deepcopy(self._node(path))
        new = _new_check(self._method(ref), self.position(path), named=not path)
        put(self._data, path, {key: old[key] for key in PLACE if key in old} | new)
        self._validated = None
        inner = {key: value for key, value in old.items() if key not in PLACE}
        for slot in self.slots(path):
            if not slot.checks and inner["uses"] in {info.ref for info in self.methods(slot.new)}:
                put(self._data, slot.path, [inner] if slot.many else inner)
                self._validated = None
                return

    def add(self, path: str, ref: str, question: QuestionId | None = None) -> None:
        """Adds a check of the method ``ref`` names at ``path``, a slot's new path, reading
        ``question`` if given."""
        slots = self.slots(_owner_of(_slot_of(path)))
        slot = next((slot for slot in slots if slot.new == path), None)
        if slot is None or slot.full:
            raise EditingError(f"{path} isn't a place for another check.")
        new = _new_check(
            self._method(ref), self.position(path, question), named=question is not None
        )
        if slot.many:
            put(self._data, slot.path, [*(get(self._data, slot.path) or []), new])
        else:
            put(self._data, slot.path, new)
        self._validated = None

    def remove(self, path: str) -> str:
        """Removes the check at ``path``, inside the marking; the path of the check that held
        it."""
        if not path or not self.is_check(path):
            raise EditingError(f"{path} isn't a check inside this marking.")
        parent, last = _parent_of(path)
        get(self._data, parent).pop(last)
        self._validated = None
        return _owner_of(_slot_of(path))

    def move(self, path: str, offset: int) -> None:
        """Moves the check at ``path`` within its list; at either end, no further."""
        parent, last = _parent_of(path) if path else ("", None)
        items = get(self._data, parent)
        if not self.is_check(path) or not isinstance(last, int) or not isinstance(items, list):
            raise EditingError(f"{path} isn't a check in a list.")
        if 0 <= (other := last + offset) < len(items):
            items[last], items[other] = items[other], items[last]
            self._validated = None

    def slots(self, path: str) -> list[Slot]:
        """Where the check at ``path`` holds checks."""
        method = self.method(path)
        if method is None:
            return []
        properties = self.schema(path)["properties"]
        found = []
        for name, info in method.settings_at(self.position(path)).model_fields.items():
            held = holds_checks(info)
            if held is None:
                continue
            at, prop = join(path, "with", name), properties.get(name, {})
            value = get(self._data, at)
            if held == "many":
                checks = tuple(join(at, index) for index in range(len(value or [])))
            else:
                checks = (at,) if value else ()
            found.append(
                Slot(
                    name=name,
                    path=at,
                    title=info.title or name,
                    many=held == "many",
                    minimum=prop.get("minItems", 0),
                    maximum=prop.get("maxItems"),
                    checks=checks,
                )
            )
        return found

    def label(self, path: str) -> str:
        """The check's name, the value of the answer it reads, or else its method's label."""
        node, position = self._node(path), self.position(path)
        if name := self._resolved(node.get("name")):
            return str(name)
        value = (
            position.value if position.value_fixed else path and self._resolved(node.get("value"))
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
                else f"Takes off {format_fraction(-weight)}"
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
                check = check_of(self._node(path), self._lookup)
            except ValidationError:
                return ""
        return check.describe() if check else ""

    def earlier(self, path: str) -> list[str]:
        """Paths of checks whose results the condition at ``path`` may read: those before it
        in its list, and in the lists holding it."""
        checks = self._checks()
        marking, found = self._validate()
        if marking is not None and not found and (target := checks.get(path)) is not None:
            preceding = _execution_earlier(marking, target, []) or []
            paths = {id(check): at for at, check in checks.items()}
            return [at for check in preceding if (at := paths.get(id(check))) is not None]
        return self._written_earlier(path)

    def _written_earlier(self, path: str) -> list[str]:
        """``earlier`` as the data is written, for a marking that isn't a check yet."""
        return [at for found in self._walk() if _within(path, found.path) for at in found.siblings]

    def finalized(self) -> Marking:
        """The marking, once nothing is wrong with it. Raises ``EditingError`` with its first
        problem."""
        marking, found = self._validate()
        if found:
            raise EditingError(str(found[0]))
        assert marking is not None  # with no problems, it is a check
        return marking

    @property
    def problems(self) -> list[Problem]:
        """What is wrong with the marking, each at the check and field at fault; as read from
        YAML, each after its line instead."""
        found = self._validate()[1]
        if self._yaml is None or self._yaml[1] != self._data:
            return found
        document = self._yaml[0]
        placed = (Problem(on_line(document, _loc(p, self._data), str(p))) for p in found)
        return list(dict.fromkeys(placed))

    @property
    def _scope(self) -> Mapping[str, Shape]:
        """What its templates may name beside earlier checks."""
        return {}

    def _resolved(self, value: Any) -> Any:
        """``value`` as the marking reads it."""
        return value

    def _readable(self, path: str) -> bool:
        """Whether a condition of the check at ``path`` has something to read."""
        return bool(self.earlier(path)) or any(self._scope.values())

    def _validate(self) -> tuple[Check | None, list[Problem]]:
        if self._validated is None:
            self._validated = self._checked(self._data)
        return self._validated

    def _checked(self, data: JsonDict) -> tuple[Check | None, list[Problem]]:
        try:
            check = check_of(data, self._lookup)
        except ValidationError as error:
            misfits = self._misfits()
            found = validation_problems(error, data, self._has_check)
            return None, [
                *misfits,
                *(p for p in found if not any(_within(p.check, m.check) for m in misfits)),
                *map(self._misplaced, self._unread()),
            ]
        return check, [
            self._misplaced(problem)
            for problem in located_problems(
                check, self.questions, scope=self._scope, is_check=self._has_check
            )
        ]

    def _unread(self) -> Iterator[Problem]:
        """The conditions naming a check not before them, said before the marking is whole."""
        for found in self._walk():
            source = get(self._data, found.path).get("if")
            if not isinstance(source, str):
                continue
            before = {self._id(at) for at in self._written_earlier(found.path)}
            for name in sorted(named(source, "checks") - before):
                yield Problem(unknown_check(name), found.path, ("if",))

    def _misfits(self) -> list[Problem]:
        """The checks whose method can't mark where they are, said before the marking is whole."""
        return [
            Problem(message, found.path)
            for found in self._walk()
            if found.method and (message := misfit(found.method, found.position))
        ]

    def _misplaced(self, problem: Problem) -> Problem:
        """A problem with a condition that reads a check after it, as what to do about it."""
        source = get(self._data, join(problem.check, "if")) if problem.field == ("if",) else None
        if not isinstance(source, str):
            return problem
        later = {
            id_: found.path
            for found in self._walk()
            if any(_within(problem.check, at) for at in found.siblings)
            and (id_ := self._id(found.path))
        }
        read = [later[name] for name in sorted(named(source, "checks")) if name in later]
        if not read:
            return problem
        return replace(problem, message=f"{self.label(read[0])} must come before this check.")

    def _has_check(self, loc: Loc) -> bool:
        return self.is_check(join(*loc))

    def _checks(self) -> dict[str, Check]:
        """The checks of the whole marking by path, once it is a check."""
        marking = self._validate()[0]
        return {join(*loc): check for loc, check in written_checks(marking)} if marking else {}

    def _node(self, path: str) -> JsonDict:
        if not self.is_check(path):
            raise _not_a_check(path)
        node: JsonDict = get(self._data, path)
        return node

    def _method(self, name: str) -> Method[Any]:
        method = self._lookup.get(name)
        if method is None:
            raise EditingError(f"There's no method {name}.")
        return method

    def _id(self, path: str) -> str | None:
        node = get(self._data, path)
        return str(node["id"]) if isinstance(node, Mapping) and node.get("id") else None

    def _condition(self, path: str, condition: Condition) -> str:
        """The condition as the check's ``if``, giving each check it reads an ID if it has none."""
        readable = set(self.earlier(path))
        rows = tuple(
            replace(row, check=self._id(row.check) or self._name(row.check))
            for row in condition.rows
            if row.check in readable
        )
        return written_condition(replace(condition, rows=rows))

    def _name(self, path: str) -> str:
        """Gives the check at ``path`` an ID made from its label."""
        taken = {id_ for at in self.paths() if (id_ := self._id(at))}
        stem = re.sub(r"\W+", "_", self.label(path).lower()).strip("_") or "check"
        stem = stem if stem[0].isalpha() else f"check_{stem}"
        name = first_free(stem, taken)
        node = {**self._node(path), "id": name}
        put(self._data, path, {key: node[key] for key in KEYS if key in node})
        return name


def _loc(problem: Problem, data: JsonDict) -> Loc:
    """Where the problem is in the YAML: a field the check doesn't have as a key is a setting,
    under its ``with``, and a problem with no field is at the check's ``uses``."""
    node, field = get(data, problem.check), problem.field
    if not field:
        return (*parts(problem.check), "uses")
    key = field[0] in KEYS or (isinstance(node, Mapping) and field[0] in node)
    return (*parts(problem.check), *(field if key else ("with", *field)))


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


def _inside(method: Method[Any], position: Position, loc: Loc) -> Position:
    """Where a check at ``loc`` of the settings of a check of ``method`` at ``position`` is,
    before its own place."""
    in_list = isinstance(loc[-1], int)
    return method.inside(position.inside(None, None, in_list=in_list), loc)


def _execution_earlier(check: Check, target: Check, earlier: list[Check]) -> list[Check] | None:
    if check is target:
        return earlier
    lists: dict[Loc, list[Check]] = {}
    for loc, child, in_list in check.method.validation_children(check):
        slot = loc[:-1] if in_list else loc
        siblings = lists.setdefault(slot, [])
        scope = [*earlier, *siblings] if check.method.ordered(dict(check.with_)) else earlier
        if (found := _execution_earlier(child, target, scope)) is not None:
            return found
        siblings.append(child)
    return None


def _new_check(method: Method[Any], position: Position, *, named: bool) -> JsonDict:
    """A new check of ``method`` at ``position``, naming its question if ``named``. Raises
    ``EditingError`` for a method that can't mark there."""
    if method.level == "internal" or not method.fits(position):
        raise EditingError(f"{method.label} can't mark here.")
    new: JsonDict = {"uses": method.ref, "with": method.initial_settings(position)}
    return {"question": position.question, **new} if named and position.question else new


def first_free(stem: str, taken: Container[object], separator: str = "_") -> str:
    """``stem``, or else the first of it numbered from 2 after ``separator`` that isn't taken."""
    numbered = (f"{stem}{separator}{n}" for n in count(2))
    return next(name for name in chain([stem], numbered) if name not in taken)


def _parent_of(path: str) -> tuple[str, str | int]:
    *parent, last = parts(path)
    return join(*parent), last


def _slot_of(path: str) -> str:
    """The slot a check in a list, like ``with.checks.0``, is in: ``with.checks``."""
    segments = parts(path)
    return join(*segments[:-1]) if segments and isinstance(segments[-1], int) else path


def _owner_of(slot: str) -> str:
    """The check a slot, like ``with.checks``, belongs to: here the marking itself, ``""``."""
    return join(*parts(slot)[:-2])


def _within(path: str, root: str) -> bool:
    return not root or path == root or path.startswith(f"{root}.")


def _check_schema(
    method: Method[Any], position: Position, holder: Position | None = None
) -> JsonDict:
    """The JSON schema of the fields of a check of ``method`` at ``position``: those that place it
    in a question group's list first, then its method's settings there, then the other keys a
    check there may have. A check in a question group's list chooses from the supplied
    questions, unless a check holding it, at ``holder``, chose its question; so with values."""
    keys = ["if_", "feedback"]
    if position.in_list:
        keys = ["name", "weight", *keys]
    # A method of several questions names its questions in its own checks.
    first = (
        ["question", "read_only"]
        if position.in_group
        and position.in_list
        and not method.group
        and (holder is None or holder.question is None)
        else []
    )
    check = Check.model_fields
    fields: dict[str, Any] = {key: (check[key].annotation, check[key]) for key in first}
    if "question" in fields and position.questions:
        # A copy without its default: merging field infos would keep the default.
        required = copy(check["question"])
        required.default = PydanticUndefined
        fields["question"] = (literal(position.questions), required)
    settings = method.settings_at(position).model_fields
    fields |= {name: (info.annotation, info) for name, info in settings.items()}
    fields |= {key: (check[key].annotation, check[key]) for key in keys}
    question = position.questions.get(position.question) if position.question else None
    # A method for values questions marks all of an answer's values, so its checks don't pick one.
    whole = method.question_types == {"values"}
    if (
        question is not None
        and question.value_count
        and position.in_list
        and not position.value_fixed
        and (holder is None or holder.value is None)
        and not whole
    ):
        fields["value"] = (literal(range(1, question.value_count + 1)) | None, check["value"])
    return json_schema(
        create_model(
            f"{method.label} check",
            __config__=ConfigDict(populate_by_name=True, title=method.label),
            **fields,
        )
    )
