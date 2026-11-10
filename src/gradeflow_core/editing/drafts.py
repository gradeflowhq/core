"""A marking being edited one step at a time, as YAML holds it."""

import re
from collections import Counter
from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from itertools import chain, count
from typing import Any, Literal, TypeGuard

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, create_model
from pydantic_core import PydanticUndefined

from ..common.documents import load_yaml_with_lines, with_line
from ..common.exceptions import EditingError, InvalidError
from ..common.hints import HINT_KEY, hint, hint_of
from ..common.paths import Location, get_at, join_path, path_parts, set_at
from ..common.problems import Problem, validation_problems
from ..common.schemas import JsonDict, json_schema
from ..common.text import format_fraction, plain_markdown
from ..methods.catalogue import BUILTINS
from ..methods.collection import Methods
from ..methods.input_sources import (
    ANSWER_INPUT,
    AnswerValueSource,
    OtherAnswerSource,
    OutputQuestions,
    StepOutputSource,
    input_source,
)
from ..methods.method import Method
from ..methods.position import Position
from ..methods.problems import (
    located_problems,
    output_question,
    step_input_position,
    unusable_reason,
    written_steps,
)
from ..methods.settings import annotated, literal_type
from ..methods.step import (
    STEP_KEYS,
    STEP_OWN_FIELDS,
    Check,
    Step,
    Transform,
    load_check,
    load_step,
    settings_of,
    step_cardinality,
)
from ..methods.templates import Shape, is_template, names_read, unknown_step_message
from ..questions.base import Question
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from .by_example import Decision, answer_decisions, mark_by_example
from .conditions import Condition, condition_template, read_condition

POSITION_KEYS = ("id", "question", "input", "name", "weight", "if")
"""The keys of a step that stay when its method changes."""
type _ReadableInput = AnswerValueSource | OtherAnswerSource | StepOutputSource
"""An input a step can read besides the answer."""
EMPTY_VALUES = (None, "")
"""What a field left empty holds."""


def compatible_methods(position: Position, *, methods: Methods = BUILTINS) -> Methods:
    """The methods a step at ``position`` can use, in their order."""
    return Methods(method for method in methods if _is_available_at(method, position))


@dataclass(frozen=True)
class StepsSetting:
    """A setting of a step that holds one step or, if ``is_list``, a list of them."""

    name: str
    path: str
    title: str
    is_list: bool
    minimum: int
    maximum: int | None
    step_paths: tuple[str, ...]
    item_label: str
    """What one entry of it is called."""

    @property
    def new_step_path(self) -> str:
        """The path a step added to it gets."""
        return join_path(self.path, len(self.step_paths)) if self.is_list else self.path

    @property
    def is_full(self) -> bool:
        if not self.is_list:
            return bool(self.step_paths)
        return self.maximum is not None and len(self.step_paths) >= self.maximum


@dataclass(frozen=True)
class _Place:
    """Where a step of a marking is, or would go."""

    path: str
    method: Method[Any] | None
    position: Position
    enclosing_position: Position
    """Where it is before its own question and input: what the steps holding it chose."""
    earlier_paths: tuple[str, ...]
    """The paths of the steps before it in its list, when they run in order."""
    output_questions: OutputQuestions
    """What it can read of the steps before it."""
    is_written: bool = True
    """Whether a step is there, rather than a place a new one can go."""


class MarkingDraft:
    """A marking of ``question``, or of a question group without one, as it is being edited. Its
    steps are at paths such as ``with.steps.0``; ``""`` is the marking itself. A step's fields are
    its method's settings and its own keys, as one object. A change to a step or setting that
    isn't there raises ``EditingError``."""

    def __init__(
        self,
        data: Mapping[str, Any],
        questions: QuestionSet,
        question: QuestionId | None,
        *,
        methods: Methods = BUILTINS,
    ) -> None:
        self.questions, self.question = questions, question
        self._methods = methods
        self._data: JsonDict = deepcopy(dict(data))
        self._schemas: dict[str, JsonDict] = {}
        self._cached_places: dict[str, _Place] | None = None
        self._cached_validation: tuple[dict[str, Step], list[Problem]] | None = None
        self._document: Any = None
        """The YAML it was read from, until it changes."""

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
        document, data = load_yaml_with_lines(text)
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
        draft._document = document
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
        draft.replace(draft._new_step_at("", draft._method(reference), None, names_question=True))
        return draft

    @property
    def data(self) -> JsonDict:
        """The marking as YAML holds it, as written."""
        return deepcopy(self._data)

    def replace(self, data: Mapping[str, Any]) -> None:
        self._put("", deepcopy(dict(data)))

    def _clear_caches(self) -> None:
        self._schemas.clear()
        self._cached_places = self._cached_validation = self._document = None

    def paths(self) -> list[str]:
        """The paths of its steps, each before those it holds."""
        return [path for path, place in self._places().items() if place.is_written]

    def is_step(self, path: str) -> bool:
        return (place := self._places().get(path)) is not None and place.is_written

    def method(self, path: str) -> Method[Any] | None:
        place = self._places().get(path)
        return place.method if place else None

    def position(self, path: str, question: QuestionId | None = None) -> Position:
        """Where the step at ``path`` is, or would be; reading ``question`` if given. Raises
        ``EditingError`` for a path no step can be at."""
        position = self._place(path).position
        return position.for_question(question) if question else position

    @property
    def _root_position(self) -> Position:
        return Position(self.questions, self.question, is_in_group=self.question is None)

    def _places(self) -> dict[str, _Place]:
        """Each step of the marking by path, each before those it holds and the places left for
        more."""
        if self._cached_places is None:
            places = self._places_within(self._data, "", self._root_position, (), {})
            self._cached_places = {place.path: place for place in places}
        return self._cached_places

    def _places_within(
        self,
        yaml_value: Any,
        path: str,
        enclosing_position: Position,
        earlier_paths: tuple[str, ...],
        output_questions: OutputQuestions,
    ) -> Iterator[_Place]:
        """The place at ``path``, holding ``yaml_value``, and each place inside it."""
        if not _is_step(yaml_value):
            yield _Place(
                path,
                None,
                enclosing_position,
                enclosing_position,
                earlier_paths,
                output_questions,
                is_written=False,
            )
            return
        method = self._methods.by_ref.get(str(yaml_value["uses"]))
        position = self._step_position(yaml_value, enclosing_position, output_questions)
        yield _Place(path, method, position, enclosing_position, earlier_paths, output_questions)
        if method is None:
            return
        settings = settings_of(yaml_value)
        reads_earlier_results = method.reads_earlier_results(settings)
        for name, field in method.settings.model_fields.items():
            cardinality = step_cardinality(field)
            value = settings.get(name)
            location = ("with", name)
            if cardinality == "one":
                inner_position = method.inner_position(position, location)
                yield from self._places_within(
                    value, join_path(path, *location), inner_position, (), output_questions
                )
            elif cardinality == "many":
                earlier_paths_in_list: tuple[str, ...] = ()
                available_output_questions = dict(output_questions)
                for index, step_data in enumerate([*_list_or_empty(value), None]):
                    item_location = (*location, index)
                    item_path = join_path(path, *item_location)
                    inner_position = method.inner_position(position, item_location)
                    yield from self._places_within(
                        step_data,
                        item_path,
                        inner_position,
                        earlier_paths_in_list,
                        dict(available_output_questions),
                    )
                    if not _is_step(step_data) or not reads_earlier_results:
                        continue
                    earlier_paths_in_list = (*earlier_paths_in_list, item_path)
                    if step_data.get("id"):
                        available_output_questions[str(step_data["id"])] = self._output_question_of(
                            step_data, inner_position, available_output_questions
                        )

    def _step_position(
        self,
        step_data: Mapping[str, Any],
        enclosing_position: Position,
        output_questions: OutputQuestions,
    ) -> Position:
        """Where the step ``step_data`` reads, inside ``enclosing_position``."""
        return step_input_position(
            self._question_id_of(step_data),
            step_data.get("input"),
            enclosing_position,
            output_questions,
        )

    def _output_question_of(
        self,
        step_data: Mapping[str, Any],
        enclosing_position: Position,
        output_questions: OutputQuestions,
    ) -> Question | None:
        """A question reading what the transform ``step_data`` gives; None for a check."""
        method = self._methods.by_ref.get(str(step_data["uses"]))
        if method is None:
            return None
        position = self._step_position(step_data, enclosing_position, output_questions)
        return output_question(method, settings_of(step_data), position, self._methods.by_ref)

    def _place(self, path: str) -> _Place:
        place = self._places().get(path)
        if place is None:
            raise _not_a_step_error(path)
        return place

    def _question_id_of(self, step_data: Mapping[str, Any]) -> str | None:
        """The question a step names for itself."""
        question_id = self._resolved_value(step_data.get("question"))
        return (
            question_id if isinstance(question_id, str) and not is_template(question_id) else None
        )

    def methods(self, path: str, question: QuestionId | None = None) -> Methods:
        """The methods a step at ``path`` can use, reading the answer as it is or another input
        it could read."""
        positions = [
            self.position(path, question),
            self._enclosing_position(path, question),
            *self._readable_inputs(path, question).values(),
        ]
        return Methods(
            method
            for method in self._methods
            if any(_is_available_at(method, position) for position in positions)
        )

    def schema(self, path: str) -> JsonDict:
        """The JSON schema of the fields of the step at ``path``."""
        if path not in self._schemas:
            self._schemas[path] = self._built_schema(path)
        return self._schemas[path]

    def _built_schema(self, path: str) -> JsonDict:
        """The JSON schema of the fields of the step at ``path``: where it sits in a question
        group's list, then its method's settings, then the step's other keys."""
        step_data, place = self._step_data(path), self._place(path)
        method, position = self._method(str(step_data["uses"])), place.position
        field_definitions: dict[str, Any] = {}
        # A method of several questions names its questions in its own steps.
        if (
            position.is_in_group
            and position.is_in_list
            and not method.marks_question_group
            and place.enclosing_position.question is None
        ):
            question_field = copy(Check.model_fields["question"])
            if position.questions:
                # A copy without its default: merging field infos would keep the default.
                question_field.default = PydanticUndefined
                question_field.annotation = literal_type(position.questions)
            field_definitions["question"] = (question_field.annotation, question_field)
        if input_choices := self._input_choices(path, method):
            labels = {ANSWER_INPUT: "The answer", **input_choices}
            input_field = Field(default=ANSWER_INPUT, json_schema_extra=hint(labels=labels))
            field_definitions["input"] = annotated(
                literal_type(labels), Check.model_fields["input"], input_field
            )
        settings = method.settings_model_at(position).model_fields
        field_definitions |= {name: (field.annotation, field) for name, field in settings.items()}
        step_model_fields = (Transform if self._is_transform(path) else Check).model_fields
        shows_condition = (
            "if" in step_data
            or bool(self.earlier_checks(path))
            or any(self._template_names.values())
        )
        fields_after_settings = [
            *(("name", "weight") if position.is_in_list else ()),
            *(("if_",) if shows_condition else ()),
            "feedback",
        ]
        field_definitions |= {
            key: (step_model_fields[key].annotation, step_model_fields[key])
            for key in fields_after_settings
            if key in step_model_fields
        }
        schema = json_schema(
            create_model(
                f"{method.label} step",
                __config__=ConfigDict(title=method.label),
                **field_definitions,
            )
        )
        for property_schema in schema["properties"].values():
            if property_schema.get(HINT_KEY, {}).get("kind") == "output_step":
                steps_path = join_path(path, "with", "steps")
                transform_paths = [
                    step_path
                    for step_path in self.paths()
                    if step_path
                    and _parent_of(step_path)[0] == steps_path
                    and self._is_transform(step_path)
                    and self._step_id(step_path)
                ]
                choices = {
                    StepOutputSource(step_id).input_text: label
                    for step_id, label in self._output_labels(transform_paths).items()
                }
                property_schema["enum"] = list(choices)
                property_schema[HINT_KEY] = {**property_schema[HINT_KEY], "labels": choices}
        return schema

    def _output_labels(self, paths: Sequence[str]) -> dict[str, str]:
        """By ID, the label of each step at ``paths``, telling it from the others."""
        labels = _numbered([self._identifying_label(path) for path in paths])
        return {
            step_id: label
            for path, label in zip(paths, labels, strict=True)
            if (step_id := self._step_id(path))
        }

    def _is_transform(self, path: str) -> bool:
        method = self.method(path)
        return method is not None and not method.gives_marks(settings_of(self._step_data(path)))

    def _input_choices(self, path: str, method: Method[Any]) -> dict[str, str]:
        """Each input the step at ``path`` can read besides the answer, with its label."""
        readable_inputs = self._readable_inputs(path)
        if not readable_inputs:
            return {}
        paths_by_id = self._paths_by_id(self._earlier_step_paths(path))
        output_labels = self._output_labels(
            [
                paths_by_id[source.step_id]
                for source in readable_inputs
                if isinstance(source, StepOutputSource)
            ]
        )
        choices = {
            source.input_text: _input_choice_label(source, output_labels)
            for source, position in readable_inputs.items()
            if method.can_run_at(position)
        }
        written_input = self._step_data(path).get("input")
        return (
            {written_input: written_input} | choices
            if isinstance(written_input, str) and written_input and written_input not in choices
            else choices
        )

    def _readable_inputs(
        self, path: str, question_id: QuestionId | None = None
    ) -> dict[_ReadableInput, Position]:
        """Each input a step at ``path`` can read besides the answer, with the position of a step
        reading it."""
        position = self._enclosing_position(path, question_id)
        if not position.is_in_workflow:
            return {}
        output_questions = self._place(path).output_questions
        source_question = (
            self.questions.get(position.question or "") if position.value is None else None
        )
        sources: list[_ReadableInput] = [
            *map(
                AnswerValueSource,
                range(1, (source_question.value_count if source_question else 0) + 1),
            ),
            *(map(OtherAnswerSource, self.questions) if position.is_in_group else ()),
            *(
                StepOutputSource(step_id)
                for step_id, output in output_questions.items()
                if output is not None
            ),
        ]
        return {source: source.input_position(position, output_questions)[0] for source in sources}

    def numbered_labels(self, paths: Iterable[str]) -> list[str]:
        """The labels of the steps at ``paths``, those labelled alike numbered in order."""
        return _numbered([self.label(path) for path in paths])

    def _identifying_label(self, path: str) -> str:
        """The step's name, or else its label and the first line of what it does."""
        label = self.label(path)
        if self._resolved_value(self._step_data(path).get("name")):
            return label
        description = plain_markdown(self.describe(path)).partition("\n")[0]
        return f"{label}: {description}" if description else label

    def _enclosing_position(self, path: str, question_id: QuestionId | None = None) -> Position:
        """Where a step at ``path`` is before reading its input."""
        place = self._place(path)
        named_question_id = question_id or (
            self._question_id_of(get_at(self._data, path)) if place.is_written else None
        )
        return place.enclosing_position.for_question(named_question_id)

    def values(self, path: str) -> JsonDict:
        """The fields of the step at ``path``, read as its place reads them; its condition as a
        ``Condition`` if made of rows."""
        step_data, method = deepcopy(self._step_data(path)), self.method(path)
        if method:
            step_data["with"] = method.settings_read_at(settings_of(step_data), self.position(path))
        values = fields_of(step_data)
        if isinstance(values.get("if"), str):
            paths = self._paths_by_id(self.earlier_checks(path))
            if (condition := read_condition(values["if"])) is not None and all(
                row.check in paths for row in condition.rows
            ):
                values["if"] = replace(
                    condition,
                    rows=tuple(replace(row, check=paths[row.check]) for row in condition.rows),
                )
        return values

    def update(self, path: str, values: Mapping[str, Any]) -> None:
        """Sets fields of the step at ``path``; those not given stay. An optional setting left
        empty is unset, and a step key left empty or at its default is left out. A ``Condition``
        names the checks it reads by their paths, which get IDs."""
        step_data = self._step_data(path)
        schema = self.schema(path)
        merged_values = {**self.values(path), **values}
        if isinstance(condition := merged_values.get("if"), Condition):
            merged_values["if"] = self._condition_template(path, condition)
        method = self._method(step_data["uses"])
        setting_fields = method.settings_model_at(self.position(path)).model_fields
        defaults = {
            key: property_schema.get("default")
            for key, property_schema in schema["properties"].items()
        }

        def is_unset(key: str, value: Any) -> bool:
            return (
                value in EMPTY_VALUES
                and key not in schema.get("required", ())
                and (
                    defaults.get(key) is None
                    or not _type_accepts(setting_fields[key].annotation, value)
                )
            )

        extra_fields = {key: value for key, value in step_data.items() if key not in STEP_KEYS}
        settings = {
            key: value
            for key, value in merged_values.items()
            if key not in (*STEP_OWN_FIELDS, *extra_fields) and not is_unset(key, value)
        }
        step_own_values = {
            key: value
            for key, value in merged_values.items()
            if key in STEP_OWN_FIELDS and value not in (*EMPTY_VALUES, defaults.get(key))
        }
        self._put_step(
            path, {**extra_fields, "uses": step_data["uses"], "with": settings, **step_own_values}
        )
        self._give_id_if_transform(path)

    def change_method(self, path: str, reference: str) -> None:
        """Gives the step at ``path`` another method, keeping its place. A new method that holds
        steps takes the old step as its first, if it can."""
        old_step_data = deepcopy(self._step_data(path))
        new_step_data = self._new_step_at(
            path, self._method(reference), None, names_question=not path
        )
        can_keep_input = _is_available_at(self._method(reference), self.position(path))
        position_keys_kept = [key for key in POSITION_KEYS if can_keep_input or key != "input"]
        self._put(
            path,
            {key: old_step_data[key] for key in position_keys_kept if key in old_step_data}
            | new_step_data,
        )
        old_inner_step_data = {
            key: value for key, value in old_step_data.items() if key not in POSITION_KEYS
        }
        for setting in self.settings_holding_steps(path):
            if not setting.step_paths and old_inner_step_data["uses"] in {
                method.ref for method in self.methods(setting.new_step_path)
            }:
                self._put(
                    setting.path, [old_inner_step_data] if setting.is_list else old_inner_step_data
                )
                break
        self._give_id_if_transform(path)

    def add(self, path: str, reference: str, question: QuestionId | None = None) -> None:
        """Adds a step of the method ``reference`` names at ``path``, a setting's
        ``new_step_path``, reading ``question`` if given."""
        settings = self.settings_holding_steps(_enclosing_step_path(path))
        setting = next((setting for setting in settings if setting.new_step_path == path), None)
        if setting is None or setting.is_full:
            raise EditingError(f"{path} isn't a place for another step.")
        new_step_data = self._new_step_at(
            path, self._method(reference), question, names_question=question is not None
        )
        self._put(
            setting.path,
            [*_list_or_empty(get_at(self._data, setting.path)), new_step_data]
            if setting.is_list
            else new_step_data,
        )
        self._give_id_if_transform(path)

    def _give_id_if_transform(self, path: str) -> None:
        if path and self._is_transform(path) and self._step_id(path) is None:
            self._give_id(path)

    def _new_step_at(
        self,
        path: str,
        method: Method[Any],
        question_id: QuestionId | None,
        *,
        names_question: bool,
    ) -> JsonDict:
        """A new step of ``method`` at ``path``, reading the latest input that fits it if the
        answer as it is doesn't, and naming its question if ``names_question``. Raises
        ``EditingError`` for a method that can't be used there."""
        inputs: list[tuple[_ReadableInput | None, Position]] = [
            (None, self.position(path, question_id)),
            *reversed(self._readable_inputs(path, question_id).items()),
            (None, self._enclosing_position(path, question_id)),
        ]
        for source, position in inputs:
            if _is_available_at(method, position):
                return {
                    **({"input": source.input_text} if source else {}),
                    **(
                        {"question": position.question}
                        if names_question and position.question
                        else {}
                    ),
                    "uses": method.ref,
                    "with": method.initial_settings(position),
                }
        raise EditingError(f"{method.label} can't be used here.")

    def remove(self, path: str) -> str:
        """Removes the step at ``path``; the path of the step that held it."""
        if not path or not self.is_step(path):
            raise EditingError(f"{path} isn't a step inside this marking.")
        parent_path, last_part = _parent_of(path)
        get_at(self._data, parent_path).pop(last_part)
        self._clear_caches()
        return _enclosing_step_path(path)

    def move(self, path: str, offset: int) -> None:
        """Moves the step at ``path`` within its list; at either end, no further."""
        parent_path, index = _parent_of(path) if path else ("", None)
        siblings = get_at(self._data, parent_path)
        if not self.is_step(path) or not isinstance(index, int) or not isinstance(siblings, list):
            raise EditingError(f"{path} isn't a step in a list.")
        siblings.insert(max(0, min(index + offset, len(siblings) - 1)), siblings.pop(index))
        self._clear_caches()

    def _put(self, path: str, value: Any) -> None:
        set_at(self._data, path, value)
        self._clear_caches()

    def _put_step(self, path: str, step_data: Mapping[str, Any]) -> None:
        """Puts ``step_data`` at ``path``, its keys in ``STEP_KEYS`` order after any others."""
        extra_fields = {key: value for key, value in step_data.items() if key not in STEP_KEYS}
        self._put(
            path, extra_fields | {key: step_data[key] for key in STEP_KEYS if key in step_data}
        )

    def settings_holding_steps(self, path: str) -> list[StepsSetting]:
        method = self.method(path)
        if method is None:
            return []
        properties = self.schema(path)["properties"]
        settings = []
        for name, field in method.settings_model_at(self.position(path)).model_fields.items():
            cardinality = step_cardinality(field)
            if cardinality is None:
                continue
            setting_path = join_path(path, "with", name)
            property_schema = properties.get(name, {})
            value = get_at(self._data, setting_path)
            if cardinality == "many":
                step_paths = tuple(
                    join_path(setting_path, index) for index in range(len(_list_or_empty(value)))
                )
            else:
                step_paths = (setting_path,) if value else ()
            settings.append(
                StepsSetting(
                    name=name,
                    path=setting_path,
                    title=field.title or name,
                    is_list=cardinality == "many",
                    minimum=property_schema.get("minItems", 0),
                    maximum=property_schema.get("maxItems"),
                    step_paths=step_paths,
                    item_label=hint_of(field).item_label or "check",
                )
            )
        return settings

    def label(self, path: str) -> str:
        """The step's name, the value of the answer it reads, or else its method's label."""
        step_data = self._step_data(path)
        if name := self._resolved_value(step_data.get("name")):
            return str(name)
        with suppress(ValueError):
            if default_label := input_source(step_data.get("input")).default_step_label():
                return default_label
        method = self.method(path)
        return method.label if method else str(step_data["uses"])

    def notes(self, path: str) -> list[str]:
        """Notes beside the step's label: its question, input, weight and condition."""
        values = self.values(path)
        notes = [str(values["question"])] if values.get("question") else []
        if input_label := self._input_label(path, values.get("input")):
            notes.append(f"Reads {input_label}")
        weight = values.get("weight", Check.model_fields["weight"].default)
        if weight == 0:
            notes.append("Gives no marks")
        elif isinstance(weight, int | float) and weight != 1:
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

    def _input_label(self, path: str, input_text: Any) -> str | None:
        """What the step at ``path`` reads, in words, if it isn't the answer as it is."""
        try:
            source = input_source(input_text)
        except ValueError:
            return None
        earlier_step_paths = self._earlier_step_paths(path)
        labels = _numbered(
            [self._identifying_label(earlier_step_path) for earlier_step_path in earlier_step_paths]
        )
        labels_by_id = dict(zip(map(self._step_id, earlier_step_paths), labels, strict=True))
        return source.input_label(lambda step_id: labels_by_id.get(step_id, step_id))

    def describe(self, path: str) -> str:
        """What the step at ``path`` does; ``""`` until it is valid."""
        step = self._validation()[0].get(path)
        if step is None and self.is_step(path):
            try:
                step = load_step(self._step_data(path), self._methods.by_ref)
            except ValidationError:
                return ""
        return step.describe() if step else ""

    def earlier_checks(self, path: str) -> list[str]:
        """Paths of checks whose results the condition at ``path`` may read: those before it
        in its list, and in the lists holding it."""
        return [
            earlier_step_path
            for earlier_step_path in self._earlier_step_paths(path)
            if not self._is_transform(earlier_step_path)
        ]

    def _earlier_step_paths(self, path: str) -> list[str]:
        """Paths of the steps before the one at ``path`` in its list, and in the lists holding
        it."""
        return [
            sibling
            for place in self._places().values()
            if place.is_written and _is_within(path, place.path)
            for sibling in place.earlier_paths
        ]

    def _paths_by_id(self, paths: Iterable[str]) -> dict[str, str]:
        """The steps at ``paths`` that have an ID, by ID; the last of those sharing one."""
        return {step_id: path for path in paths if (step_id := self._step_id(path))}

    def answer_decisions(self) -> dict[str, Decision]:
        """The answers given marks by example, with what each gets.

        Keys use the question's spelling. Other methods' matches are not decisions. A decision
        describes the explicit rule, independently of conditions and other methods' results.
        """
        return answer_decisions(self._data, self._methods, self._question)

    def mark_by_example(
        self,
        answers: Sequence[str],
        decision: Decision | Literal["cancel"],
        proportion: float = 0.5,
    ) -> None:
        """Gives these answers one of the ``DECISIONS``, ``proportion`` of the marks for
        ``"partial_marks"``, or takes their decision back with ``"cancel"``. Other answers keep
        theirs.

        A decision overrides the existing marking. Cancelling leaves the remaining methods to mark
        the answers; an answer must be cancelled before it gets another decision, and giving one
        the decision it has changes nothing. Equivalent spellings of the question's answers are
        one answer; its type must be one that can be marked by example. New checks use the
        highest version of each method.
        """
        marking = mark_by_example(
            self._data, answers, decision, proportion, self._methods, self._question
        )
        if marking is not None:
            self.replace(marking)

    @property
    def _question(self) -> Question | None:
        return self.questions[self.question] if self.question else None

    def finalized(self) -> Check:
        """The marking, once nothing is wrong with it. Raises ``EditingError`` with its first
        problem."""
        steps, problems = self._validation()
        if problems:
            raise EditingError(str(problems[0]))
        marking = steps[""]
        assert isinstance(marking, Check)  # a method's steps may give a value instead
        return marking

    @property
    def problems(self) -> list[Problem]:
        """What is wrong with the marking, each at the step and field at fault; as read from
        YAML, each after its line instead."""
        problems = self._validation()[1]
        if self._document is None:
            return problems
        problems_with_lines = (
            Problem(with_line(self._document, _yaml_location(problem, self._data), str(problem)))
            for problem in problems
        )
        return list(dict.fromkeys(problems_with_lines))

    @property
    def _template_names(self) -> Mapping[str, Shape]:
        """What its templates may name besides earlier checks."""
        return {}

    def _resolved_value(self, value: Any) -> Any:
        """``value`` as the marking reads it."""
        return value

    def _validation(self) -> tuple[dict[str, Step], list[Problem]]:
        """The marking's steps by path, none until it is valid, and its problems."""
        if self._cached_validation is None:
            marking, problems = self._validation_of(self._data)
            steps = (
                {join_path(*location): step for location, step in written_steps(marking)}
                if marking
                else {}
            )
            self._cached_validation = steps, problems
        return self._cached_validation

    def _load_root(self, data: JsonDict) -> Step:
        """The root step ``data`` holds."""
        return load_check(data, self._methods.by_ref)

    def _validation_of(self, data: JsonDict) -> tuple[Step | None, list[Problem]]:
        try:
            step = self._load_root(data)
        except ValidationError as error:
            position_problems = self._position_problems()
            problems = validation_problems(error, data, self._is_step_at)
            return None, [
                *position_problems,
                *(
                    problem
                    for problem in problems
                    if not any(
                        _is_within(problem.step_path, position_problem.step_path)
                        for position_problem in position_problems
                    )
                ),
                *map(self._condition_order_problem, self._unknown_step_problems()),
            ]
        return step, [
            self._condition_order_problem(problem)
            for problem in located_problems(
                step, self.questions, scope=self._template_names, is_step_at=self._is_step_at
            )
        ]

    def _unknown_step_problems(self) -> Iterator[Problem]:
        """A problem at each condition naming a check that isn't before it."""
        for place in self._places().values():
            condition = (
                get_at(self._data, join_path(place.path, "if")) if place.is_written else None
            )
            if not isinstance(condition, str):
                continue
            earlier_ids = self._paths_by_id(self.earlier_checks(place.path)).keys()
            for step_id in sorted(names_read(condition, "steps") - earlier_ids):
                yield Problem(unknown_step_message(step_id), place.path, ("if",))

    def _position_problems(self) -> list[Problem]:
        """A problem at each step whose method can't run where it is."""
        return [
            Problem(message, place.path)
            for place in self._places().values()
            if place.method and (message := unusable_reason(place.method, place.position))
        ]

    def _condition_order_problem(self, problem: Problem) -> Problem:
        """``problem``, naming the check to move if its condition reads a later one."""
        condition = (
            get_at(self._data, join_path(problem.step_path, "if"))
            if problem.field == ("if",)
            else None
        )
        if not isinstance(condition, str):
            return problem
        later_paths_by_id = {
            step_id: place.path
            for place in self._places().values()
            if place.is_written
            and any(_is_within(problem.step_path, sibling) for sibling in place.earlier_paths)
            and (step_id := self._step_id(place.path))
        }
        later_checks_read = [
            later_paths_by_id[step_id]
            for step_id in sorted(names_read(condition, "steps"))
            if step_id in later_paths_by_id
        ]
        if not later_checks_read:
            return problem
        label = self.label(later_checks_read[0])
        return replace(problem, message=f"{label} must come before this step.")

    def _is_step_at(self, location: Location) -> bool:
        return self.is_step(join_path(*location))

    def _step_data(self, path: str) -> JsonDict:
        """The step at ``path`` as YAML holds it."""
        if not self.is_step(path):
            raise _not_a_step_error(path)
        step_data: JsonDict = get_at(self._data, path)
        return step_data

    def _method(self, reference: str) -> Method[Any]:
        method = self._methods.by_ref.get(reference)
        if method is None:
            raise EditingError(f"There's no method {reference}.")
        return method

    def _step_id(self, path: str) -> str | None:
        step_id = get_at(self._data, path).get("id")
        return str(step_id) if step_id else None

    def _condition_template(self, path: str, condition: Condition) -> str:
        """The condition as the check's ``if``, giving each check it reads an ID if it has none."""
        earlier_check_paths = set(self.earlier_checks(path))
        rows = tuple(
            replace(row, check=self._step_id(row.check) or self._give_id(row.check))
            for row in condition.rows
            if row.check in earlier_check_paths
        )
        return condition_template(replace(condition, rows=rows))

    def _give_id(self, path: str) -> str:
        """Gives the step at ``path`` an ID made from its label."""
        taken = {step_id for step_path in self.paths() if (step_id := self._step_id(step_path))}
        stem = re.sub(r"\W+", "_", self.label(path).lower()).strip("_") or "step"
        stem = stem if stem[0].isalpha() else f"step_{stem}"
        new_step_id = first_unused_name(stem, taken)
        self._put_step(path, {**self._step_data(path), "id": new_step_id})
        return new_step_id


def _yaml_location(problem: Problem, data: JsonDict) -> Location:
    """Where the problem is in the YAML."""
    step_data, field = get_at(data, problem.step_path), problem.field
    if not field:
        return (*path_parts(problem.step_path), "uses")
    is_step_key = field[0] in STEP_KEYS or (
        isinstance(step_data, Mapping) and field[0] in step_data
    )
    return (*path_parts(problem.step_path), *(field if is_step_key else ("with", *field)))


def _type_accepts(annotation: Any, value: Any) -> bool:
    """Whether ``value`` is of type ``annotation``, ignoring its constraints."""
    try:
        TypeAdapter[Any](annotation).validate_python(value)
    except ValidationError:
        return False
    return True


def _not_a_step_error(path: str) -> EditingError:
    return EditingError(f"{path} isn't a step of this marking.")


def fields_of(step_data: Mapping[str, Any]) -> JsonDict:
    """A step's settings and keys as one."""
    return {
        **settings_of(step_data),
        **{key: value for key, value in step_data.items() if key not in ("uses", "with")},
    }


def field_path(step_path: str, key: str) -> str:
    """The path of the field ``key`` of the step at ``step_path``."""
    return (
        join_path(step_path, key) if key in STEP_OWN_FIELDS else join_path(step_path, "with", key)
    )


def _is_step(yaml_value: Any) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(yaml_value, Mapping) and "uses" in yaml_value


def _is_available_at(method: Method[Any], position: Position) -> bool:
    return (
        method.level != "internal"
        and method.can_run_at(position)
        and (method.gives_marks({}) or position.is_in_workflow)
    )


def first_unused_name(stem: str, taken: Container[object], separator: str = "_") -> str:
    """``stem``, or else the first of it numbered from 2 after ``separator`` that isn't taken."""
    numbered_names = (f"{stem}{separator}{number}" for number in count(2))
    return next(name for name in chain([stem], numbered_names) if name not in taken)


def _parent_of(path: str) -> tuple[str, str | int]:
    *parent_parts, last_part = path_parts(path)
    return join_path(*parent_parts), last_part


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _enclosing_step_path(path: str) -> str:
    """The path of the step holding the step at ``path``, as ``with.steps.0`` holds
    ``with.steps.0.with.steps.1`` and ``with.steps.0.with.check``."""
    parts = path_parts(path)
    is_in_list = bool(parts) and isinstance(parts[-1], int)
    return join_path(*parts[: -3 if is_in_list else -2])


def _is_within(path: str, ancestor_path: str) -> bool:
    return not ancestor_path or path == ancestor_path or path.startswith(f"{ancestor_path}.")


def _numbered(labels: Sequence[str]) -> list[str]:
    """``labels``, those alike numbered in order."""
    label_counts_so_far: Counter[str] = Counter()
    numbered_labels = []
    for label in labels:
        label_counts_so_far[label] += 1
        numbered_labels.append(
            f"{label} {label_counts_so_far[label]}" if labels.count(label) > 1 else label
        )
    return numbered_labels


def _input_choice_label(source: _ReadableInput, output_labels: Mapping[str, str]) -> str:
    """How an input a step can choose reads, naming steps by ``output_labels``."""
    match source:
        case AnswerValueSource(value_number=value_number):
            return f"Value {value_number}"
        case OtherAnswerSource(question_id=question_id):
            return question_id
        case StepOutputSource(step_id=step_id):
            return f"Output of {output_labels[step_id]}"
