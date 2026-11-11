"""A marking draft of a method's steps, whose fields may be bound to its declared settings."""

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from pydantic.fields import FieldInfo

from ..common.exceptions import EditingError
from ..common.paths import value_at
from ..common.problems import Problem
from ..common.schemas import JsonDict
from ..methods.collection import Methods
from ..methods.composed.file import TEMPLATE_KEYS, WORKFLOW_REF
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from ..steps.position import Position
from ..steps.settings import step_cardinality
from ..steps.step import STEP_OWN_PROPERTIES
from ..steps.templates import Shape, is_template, reference_expression
from .drafts import MarkingDraft, field_operation, field_path, fields_of


class MethodStepsDraft(MarkingDraft):
    """A method's steps, edited as one Workflow. ``declared_settings`` is shared with the
    ``MethodDraft`` that owns it."""

    _must_give_marks = False

    def __init__(
        self,
        data: Mapping[str, Any],
        questions: QuestionSet,
        question: QuestionId | None,
        methods: Methods,
        declared_settings: dict[str, dict[str, Any]],
        test_values: Mapping[str, Any],
    ) -> None:
        self._declared_settings = declared_settings
        self._test_values = dict(test_values)
        super().__init__(data, questions, question, methods=methods)

    @property
    def test_values(self) -> dict[str, Any]:
        return self._test_values

    @test_values.setter
    def test_values(self, values: Mapping[str, Any]) -> None:
        self._test_values = dict(values)
        self._set_data(self._data)

    def set_declared_setting(self, name: str, declaration: dict[str, Any]) -> None:
        """Declare the setting ``name``, or replace its declaration."""
        self._declared_settings[name] = declaration
        self._set_data(self._data)

    def remove_declared_setting(self, name: str) -> None:
        self._declared_settings.pop(name, None)
        self._test_values.pop(name, None)
        self._set_data(self._data)

    @property
    def _root_position(self) -> Position:
        return replace(super()._root_position, is_in_workflow=True)

    def methods(self, path: str, question: QuestionId | None = None) -> Methods:
        return (
            Methods([self._method(WORKFLOW_REF)]) if not path else super().methods(path, question)
        )

    def schema(self, path: str) -> JsonDict:
        schema = super().schema(path)
        if path:
            return schema
        properties = schema["properties"].items()
        return {
            **schema,
            "properties": {
                key: value for key, value in properties if key not in STEP_OWN_PROPERTIES
            },
        }

    def values(self, path: str) -> dict[str, Any]:
        return {
            name: value if name in TEMPLATE_KEYS else self._resolved_value(value)
            for name, value in super().values(path).items()
        }

    def bound_setting(self, value: Any) -> str | None:
        """The declared setting ``value`` names, as in ``{{ settings.minimum }}``."""
        match = re.fullmatch(r"settings\.(\w+)", reference_expression(value) or "")
        return match[1] if match and match[1] in self._declared_settings else None

    def _resolved_value(self, value: Any, test_values: Mapping[str, Any] = {}) -> Any:
        """``value``, or the default of the declared setting it is bound to, else its value in
        ``test_values``."""
        name = self.bound_setting(value)
        if name is None:
            return deepcopy(value)
        return deepcopy(self._declared_settings[name].get("default", test_values.get(name)))

    def update(self, path: str, values: Mapping[str, Any]) -> None:
        """Sets the step's fields; a bound field sets its declared setting's default instead."""
        old_step_data = self._step_data(path)
        fields = fields_of(old_step_data)
        bound_settings_by_key = {
            key: name
            for key, value in fields.items()
            if key not in (*TEMPLATE_KEYS, "for_each") and (name := self.bound_setting(value))
        }
        defaults: dict[str, Any] = {}
        for key, name in bound_settings_by_key.items():
            if key not in values or values[key] == self._resolved_value(fields[key]):
                continue
            if name in defaults and defaults[name] != values[key]:
                raise EditingError(f"Fields sharing setting {name} must have the same default.")
            defaults[name] = values[key]
        super().update(path, values)
        step_data = self._step_data(path)
        for name, value in defaults.items():
            self._declared_settings[name].pop("default", None)
            if not is_empty(value):
                self._declared_settings[name]["default"] = deepcopy(value)
        self._patch(
            *(field_operation(path, step_data, key, fields[key]) for key in bound_settings_by_key)
        )

    @property
    def _template_names(self) -> Mapping[str, Shape]:
        return {"settings": dict.fromkeys(self._declared_settings, "any")}

    @property
    def problems(self) -> list[Problem]:
        """What is wrong with the steps, leaving out fields a declared setting fills."""
        return [problem for problem in super().problems if not self._is_filled_by_setting(problem)]

    def _written_data(self, data: JsonDict) -> JsonDict:
        written_data: JsonDict = self._with_resolved_questions(data)
        return written_data

    def _marking_data(self, data: JsonDict) -> JsonDict:
        marking_data: JsonDict = self._resolved_data(data)
        return marking_data

    def _with_resolved_questions(self, value: Any) -> Any:
        """``value`` with the question each step names filled in from its setting's default, and
        left out while a template says it."""
        if isinstance(value, list):
            return [self._with_resolved_questions(item) for item in value]
        if not isinstance(value, dict):
            return value
        data = {key: self._with_resolved_questions(item) for key, item in value.items()}
        if "uses" in value and "question" in value:
            question_id = self._resolved_value(value["question"])
            is_known = isinstance(question_id, str) and not is_template(question_id)
            data["question"] = question_id if is_known else None
        return data

    def _is_filled_by_setting(self, problem: Problem) -> bool:
        if not problem.field or not self.is_step(problem.step_path):
            return False
        key, method = str(problem.field[0]), self.method(problem.step_path)
        if not can_setting_fill(key, method.settings.model_fields if method else {}):
            return False
        return (
            self.bound_setting(value_at(self._data, field_path(problem.step_path, key))) is not None
        )

    def _resolved_data(self, value: Any) -> Any:
        """``value`` with each bound field filled in: its default, or else its test value."""
        if isinstance(value, dict):
            return {
                key: item if key in TEMPLATE_KEYS else self._resolved_data(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._resolved_data(item) for item in value]
        return self._resolved_value(value, self.test_values)


def can_setting_fill(key: str, settings: Mapping[str, FieldInfo]) -> bool:
    """Whether a declared setting can fill the field ``key`` of a step with these settings."""
    return key not in TEMPLATE_KEYS and (
        key not in settings or step_cardinality(settings[key]) is None
    )


def is_empty(value: Any) -> bool:
    return value in (None, "", [], {})
