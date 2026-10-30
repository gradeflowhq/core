"""A marking draft of a method's checks, whose fields may be bound to its inputs."""

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from pydantic.fields import FieldInfo

from ..common.exceptions import EditingError
from ..common.paths import get, put
from ..common.problems import Problem
from ..common.schemas import JsonDict
from ..methods.check import CHECK_FIELDS, Check, holds_checks
from ..methods.collection import Methods
from ..methods.composed.file import COMBINATION, TEMPLATE_KEYS
from ..methods.templates import Shape, is_template, reference
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from .drafts import MarkingDraft, field_path, fields_of


class MethodChecksDraft(MarkingDraft):
    """A method's checks, as the Combination of them: its settings are the file's ``checks`` and
    ``combine``, so it keeps its method and has no fields of a check. A bound field's value is its
    input's default, and changing it changes the default.
    ``inputs`` are the method's input declarations, shared with the draft that owns them."""

    def __init__(
        self,
        data: Mapping[str, Any],
        questions: QuestionSet,
        question: QuestionId | None,
        methods: Methods,
        inputs: dict[str, dict[str, Any]],
    ) -> None:
        super().__init__(data, questions, question, methods=methods)
        self._inputs = inputs

    def set_input(self, name: str, spec: dict[str, Any]) -> None:
        """Declare the input ``name``, or replace its declaration."""
        self._inputs[name] = spec
        self._validated = None

    def drop_input(self, name: str) -> None:
        self._inputs.pop(name, None)
        self._validated = None

    def methods(self, path: str, question: QuestionId | None = None) -> Methods:
        return Methods([self._method(COMBINATION)]) if not path else super().methods(path, question)

    def schema(self, path: str) -> JsonDict:
        schema = super().schema(path)
        if path:
            return schema
        properties = schema["properties"].items()
        return {**schema, "properties": {k: v for k, v in properties if k not in CHECK_FIELDS}}

    def _resolved(self, value: Any) -> Any:
        """``value``, or the default of the input it is bound to."""
        name = input_reference(value)
        if name is None or name not in self._inputs:
            return deepcopy(value)
        return deepcopy(self._inputs[name].get("default"))

    def _place(self, node: Mapping[str, Any]) -> tuple[Any, Any]:
        question, value = self._resolved(node.get("question")), self._resolved(node.get("value"))
        return (
            question if isinstance(question, str) and not is_template(question) else None,
            value if isinstance(value, int) else None,
        )

    def values(self, path: str) -> dict[str, Any]:
        values = super().values(path)
        for name, value in values.items():
            if name not in TEMPLATE_KEYS:
                values[name] = self._resolved(value)
        return values

    def update(self, path: str, values: Mapping[str, Any]) -> None:
        """Sets the check's fields; a bound field sets its input's default instead."""
        old = self._node(path)
        fields = fields_of(old)
        bound = {
            key: name
            for key, value in fields.items()
            if key not in TEMPLATE_KEYS and (name := input_reference(value)) in self._inputs
        }
        defaults: dict[str, Any] = {}
        for key, name in bound.items():
            if key not in values or values[key] == self._resolved(fields[key]):
                continue
            if name in defaults and defaults[name] != values[key]:
                raise EditingError(f"Fields sharing input {name} must have the same default.")
            defaults[name] = values[key]
        for name, value in defaults.items():
            self._inputs[name].pop("default", None)
            if not empty(value):
                self._inputs[name]["default"] = deepcopy(value)
        super().update(path, values)
        node = self._node(path)
        for key in bound:
            put(node, field_path("", key), fields[key])

    @property
    def _scope(self) -> Mapping[str, Shape]:
        return {"inputs": dict.fromkeys(self._inputs, "any")}

    @property
    def problems(self) -> list[Problem]:
        """What is wrong with the checks but their fields inputs fill, which are the method's."""
        return [problem for problem in super().problems if not self._filled(problem)]

    def _checked(self, data: JsonDict) -> tuple[Check | None, list[Problem]]:
        return super()._checked(self._resolve(data))

    def _filled(self, problem: Problem) -> bool:
        if not problem.field or not self.is_check(problem.check):
            return False
        key, method = str(problem.field[0]), self.method(problem.check)
        if not fillable(key, method.settings.model_fields if method else {}):
            return False
        return input_reference(get(self._data, field_path(problem.check, key))) in self._inputs

    def _resolve(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: item if key in TEMPLATE_KEYS else self._resolve(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        return self._resolved(value)


def fillable(key: str, settings: Mapping[str, FieldInfo]) -> bool:
    """Whether an input can fill the field ``key`` of a check with these settings: one that is
    neither a template nor holds checks."""
    return key not in TEMPLATE_KEYS and (key not in settings or holds_checks(settings[key]) is None)


def input_reference(value: Any) -> str | None:
    """The input ``value`` names, as in ``{{ inputs.minimum }}``."""
    match = re.fullmatch(r"inputs\.(\w+)", reference(value) or "")
    return match[1] if match else None


def empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}
