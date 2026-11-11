"""Steps being edited one at a time, as YAML holds them: what a marking and a method share."""

import re
from collections import Counter
from collections.abc import Container, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import chain, count
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ..common.exceptions import EditingError
from ..common.hints import HINT_KEY, hint, hint_of
from ..common.paths import (
    Location,
    is_within,
    json_pointer,
    location_of,
    patched,
    value_at,
)
from ..common.problems import Problem
from ..common.schemas import JsonDict, json_schema
from ..common.text import format_fraction, plain_markdown
from ..methods.catalogue import BUILTINS
from ..methods.collection import Methods
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from ..steps.input_sources import (
    ANSWER_INPUT,
    AnswerValueSource,
    OtherAnswerSource,
    StepOutputSource,
    input_text,
)
from ..steps.method import Method
from ..steps.places import StepPlace, marking_position, places
from ..steps.position import Position
from ..steps.problems import located_problems
from ..steps.settings import literal_type, step_cardinality
from ..steps.step import (
    CHECK_PROPERTIES,
    STEP_OWN_PROPERTIES,
    STEP_PROPERTIES,
    OutcomeFeedback,
    Step,
    load_step,
    settings_of,
)
from ..steps.templates import Shape, names_read
from .conditions import Condition, condition_template, read_condition

POSITION_KEYS = ("id", "question", "input", "name", "weight", "if")
"""The properties of a step that stay when its method changes."""
type _ReadableInput = AnswerValueSource | OtherAnswerSource | StepOutputSource
"""An input a step can read besides the answer."""
EMPTY_VALUES = (None, "")
"""What a field left empty holds."""
_STEP_SCHEMA = json_schema(Step)


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
        return f"{self.path}/{len(self.step_paths)}" if self.is_list else self.path

    @property
    def is_full(self) -> bool:
        if not self.is_list:
            return bool(self.step_paths)
        return self.maximum is not None and len(self.step_paths) >= self.maximum


@dataclass(frozen=True)
class DraftSnapshot:
    """A draft's marking as it stands, and what is worked out from it once. A change to the draft
    makes another."""

    data: JsonDict
    """The marking as YAML holds it, as written."""
    written_data: JsonDict
    """The marking as its steps are found in it."""
    marking_data: JsonDict
    """The marking as it marks."""
    methods: Methods
    position: Position
    """Where the marking is."""
    template_names: Mapping[str, Shape]
    """What its templates may name besides earlier checks."""
    document: Any = None
    """The YAML it was read from, if it hasn't changed since."""
    schemas: dict[str, JsonDict] = field(default_factory=dict)
    """The JSON schema of the fields of each step asked for, by path."""

    @cached_property
    def places(self) -> dict[str, StepPlace]:
        """The places of the steps as written."""
        return self._places_of(self.written_data)

    @cached_property
    def marking_places(self) -> dict[str, StepPlace]:
        """The places of the steps as they mark, which say what is wrong with them."""
        if self.marking_data is self.written_data:
            return self.places
        return self._places_of(self.marking_data)

    @cached_property
    def problems(self) -> list[Problem]:
        """What is wrong with the marking, each at the innermost step it writes."""
        marking = self.marking_places[""].step
        if marking is None:
            return [Problem("Choose a method.", field=("uses",))]
        problems = located_problems(marking, self.position, scope=self.template_names)
        return list(map(self._at_written_step, problems))

    def is_step(self, path: str) -> bool:
        return (place := self.places.get(path)) is not None and place.step is not None

    def _places_of(self, data: JsonDict) -> dict[str, StepPlace]:
        """Each step of the marking ``data`` holds by path, each before those it holds and the
        places left for more. Until it names a method, its only place is for one."""
        marking = load_step(data, self.methods.by_ref) if "uses" in data else None
        return {json_pointer(place.location): place for place in places(marking, self.position)}

    def _at_written_step(self, problem: Problem) -> Problem:
        """``problem`` at the innermost step the marking writes: one in a step that only a setting
        fills in is a problem of the field holding it."""
        parts = location_of(problem.step_path)
        written_length = next(
            (
                length
                for length in range(len(parts), 0, -1)
                if self.is_step(json_pointer(parts[:length]))
            ),
            0,
        )
        if written_length == len(parts):
            return problem
        is_setting = bool(problem.field) and problem.field[0] not in STEP_PROPERTIES
        location = (*parts[written_length:], *(("with",) if is_setting else ()), *problem.field)
        return replace(
            problem,
            step_path=json_pointer(parts[:written_length]),
            field=location[1:] if location[0] == "with" else location,
        )


class Draft:
    """Steps of ``question``, or of a question group without one, as they are being edited. Its
    steps are at JSON Pointers such as ``/with/steps/0``; ``""`` is the outermost. A step's fields
    are its method's settings and its own properties, as one object. A change to a step or setting
    that isn't there raises ``EditingError``."""

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
        self._set_data(deepcopy(dict(data)))

    @property
    def data(self) -> JsonDict:
        """The marking as YAML holds it, as written."""
        return deepcopy(self._data)

    @property
    def _data(self) -> JsonDict:
        return self._snapshot.data

    def replace(self, data: Mapping[str, Any]) -> None:
        self._set_data(deepcopy(dict(data)))

    def _set_data(self, data: JsonDict) -> None:
        """Makes ``data``, which nothing changes after, the marking."""
        self._snapshot = DraftSnapshot(
            data,
            self._written_data(data),
            self._marking_data(data),
            self._methods,
            self._root_position,
            self._template_names,
        )

    def paths(self) -> list[str]:
        """The paths of its steps, each before those it holds."""
        return [path for path, place in self._snapshot.places.items() if place.step is not None]

    def is_step(self, path: str) -> bool:
        return self._snapshot.is_step(path)

    def method(self, path: str) -> Method | None:
        place = self._snapshot.places.get(path)
        return place.step.resolution.method if place and place.step else None

    def position(self, path: str, question: QuestionId | None = None) -> Position:
        """Where the step at ``path`` is, or would be; reading ``question`` if given. Raises
        ``EditingError`` for a path no step can be at."""
        position = self._place(path).position
        return position.for_question(question) if question else position

    @property
    def _root_position(self) -> Position:
        return marking_position(self.questions, self.question)

    def _written_data(self, data: JsonDict) -> JsonDict:
        """The marking ``data`` holds as its steps are found in it."""
        return data

    def _marking_data(self, data: JsonDict) -> JsonDict:
        """The marking ``data`` holds as it marks."""
        return data

    def _place(self, path: str) -> StepPlace:
        place = self._snapshot.places.get(path)
        if place is None:
            raise _not_a_step_error(path)
        return place

    def _step(self, path: str) -> Step:
        """The step written at ``path``."""
        step = self._place(path).step
        if step is None:
            raise _not_a_step_error(path)
        return step

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
        schemas = self._snapshot.schemas
        if path not in schemas:
            schemas[path] = self._built_schema(path)
        return schemas[path]

    def _built_schema(self, path: str) -> JsonDict:
        """The JSON schema of the fields of the step at ``path``: where it sits in a question
        group's list, then its method's settings, then the step's other properties."""
        step_data, place = self._step_data(path), self._place(path)
        method, position = self._method(str(step_data["uses"])), place.position
        settings_schema = method.settings_schema(position)
        properties: JsonDict = {}
        required: list[str] = []
        # A method of several questions names its questions in its own steps.
        if (
            position.is_in_group
            and position.is_in_list
            and not method.marks_question_group
            and place.enclosing_position.question is None
        ):
            properties["question"] = _step_property("question", position.questions or None)
            if position.questions:
                required.append("question")
        if input_choices := self._input_choices(path, method):
            labels = {ANSWER_INPUT: "The answer", **input_choices}
            properties["input"] = {
                **_step_property("input", labels),
                "default": ANSWER_INPUT,
                **hint(labels=labels),
            }
        properties |= settings_schema["properties"]
        required += settings_schema.get("required", [])
        shows_condition = (
            "if" in step_data
            or bool(self.earlier_checks(path))
            or any(self._template_names.values())
        )
        is_transform = self._is_transform(path)
        properties |= {
            key: _step_property(key)
            for key in (
                *(("name", "weight") if position.is_in_list else ()),
                *(("if",) if shows_condition else ()),
                "feedback",
            )
            if not (is_transform and key in CHECK_PROPERTIES)
        }
        for property_schema in properties.values():
            if property_schema.get(HINT_KEY, {}).get("kind") == "output_step":
                steps_location = (*self._place(path).location, "with", "steps")
                transform_paths = [
                    step_path
                    for step_path in self.paths()
                    if location_of(step_path)[:-1] == steps_location
                    and self._is_transform(step_path)
                    and self._step_id(step_path)
                ]
                choices = {
                    input_text(StepOutputSource(step_id)): label
                    for step_id, label in self._output_labels(transform_paths).items()
                }
                property_schema["enum"] = list(choices)
                property_schema[HINT_KEY] = {**property_schema[HINT_KEY], "labels": choices}
        definitions = settings_schema.get("$defs", {})
        if "feedback" in properties:
            feedback_definition = OutcomeFeedback.__name__
            definitions[feedback_definition] = deepcopy(_STEP_SCHEMA["$defs"][feedback_definition])
        return {
            **({"$defs": definitions} if definitions else {}),
            "properties": properties,
            **({"required": required} if required else {}),
            "title": method.label,
            "type": "object",
        }

    def _output_labels(self, paths: Sequence[str]) -> dict[str, str]:
        """By ID, the label of each step at ``paths``, telling it from the others."""
        labels = _numbered([self._identifying_label(path) for path in paths])
        return {
            step_id: label
            for path, label in zip(paths, labels, strict=True)
            if (step_id := self._step_id(path))
        }

    def _is_transform(self, path: str) -> bool:
        return not self._step(path).resolution.gives_marks

    def _input_choices(self, path: str, method: Method) -> dict[str, str]:
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
            input_text(source): _input_choice_label(source, output_labels)
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
        return {source: position.reading(source, output_questions)[0] for source in sources}

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
        named_question_id = question_id or (place.step.question if place.step else None)
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
        empty is unset, and a step's own property left empty or at its default is left out. A
        ``Condition`` names the checks it reads by their paths, which get IDs."""
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

        extra_fields = {
            key: value for key, value in step_data.items() if key not in STEP_PROPERTIES
        }
        settings = {
            key: value
            for key, value in merged_values.items()
            if key not in (*STEP_OWN_PROPERTIES, *extra_fields) and not is_unset(key, value)
        }
        step_own_values = {
            key: value
            for key, value in merged_values.items()
            if key in STEP_OWN_PROPERTIES and value not in (*EMPTY_VALUES, defaults.get(key))
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
        kept_data = {key: old_step_data[key] for key in position_keys_kept if key in old_step_data}
        self._patch({"op": "replace", "path": path, "value": kept_data | new_step_data})
        old_inner_step_data = {
            key: value for key, value in old_step_data.items() if key not in POSITION_KEYS
        }
        for setting in self.settings_holding_steps(path):
            if not setting.step_paths and old_inner_step_data["uses"] in {
                method.ref for method in self.methods(setting.new_step_path)
            }:
                inner_steps = [old_inner_step_data] if setting.is_list else old_inner_step_data
                self._patch({"op": "add", "path": setting.path, "value": inner_steps})
                break
        self._give_id_if_transform(path)

    def add(self, path: str, reference: str, question: QuestionId | None = None) -> None:
        """Adds a step of the method ``reference`` names at ``path``, a setting's
        ``new_step_path``, reading ``question`` if given."""
        place = self._snapshot.places.get(path)
        step_path = _enclosing_step_path(place.location if place else ())
        settings = self.settings_holding_steps(step_path)
        setting = next((setting for setting in settings if setting.new_step_path == path), None)
        if setting is None or setting.is_full:
            raise EditingError(f"{path} isn't a place for another step.")
        new_step_data = self._new_step_at(
            path, self._method(reference), question, names_question=question is not None
        )
        steps = (
            [*_list_or_empty(value_at(self._data, setting.path)), new_step_data]
            if setting.is_list
            else new_step_data
        )
        self._patch(field_operation(step_path, self._step_data(step_path), setting.name, steps))
        self._give_id_if_transform(path)

    def _give_id_if_transform(self, path: str) -> None:
        if path and self._is_transform(path) and self._step_id(path) is None:
            self._give_id(path)

    def _new_step_at(
        self,
        path: str,
        method: Method,
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
                    **({"input": input_text(source)} if source else {}),
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
        step_path = _enclosing_step_path(self._place(path).location)
        self._patch({"op": "remove", "path": path})
        return step_path

    def move(self, path: str, offset: int) -> None:
        """Moves the step at ``path`` within its list; at either end, no further."""
        location = self._place(path).location if self.is_step(path) else ()
        if not location or not isinstance(index := location[-1], int):
            raise EditingError(f"{path} isn't a step in a list.")
        last_index = len(value_at(self._data, json_pointer(location[:-1]))) - 1
        new_location = (*location[:-1], max(0, min(index + offset, last_index)))
        self._patch({"op": "move", "from": path, "path": json_pointer(new_location)})

    def _patch(self, *operations: JsonDict) -> None:
        """Applies the JSON Patch ``operations`` to the marking."""
        self._set_data(patched(self._data, operations))

    def _put_step(self, path: str, step_data: Mapping[str, Any]) -> None:
        """Puts ``step_data`` at ``path``, its properties in their order after anything else."""
        extra_fields = {
            key: value for key, value in step_data.items() if key not in STEP_PROPERTIES
        }
        properties = {key: step_data[key] for key in STEP_PROPERTIES if key in step_data}
        self._patch({"op": "replace", "path": path, "value": extra_fields | properties})

    def settings_holding_steps(self, path: str) -> list[StepsSetting]:
        if not self.is_step(path) or (method := self.method(path)) is None:
            return []
        properties = self.schema(path)["properties"]
        inner_steps = self._step(path).resolution.inner_steps
        settings = []
        for name, field_info in method.settings_model_at(self.position(path)).model_fields.items():
            cardinality = step_cardinality(field_info)
            if cardinality is None:
                continue
            setting_path = field_path(path, name)
            property_schema = properties.get(name, {})
            if cardinality == "many":
                # The last place of a list is where another step can be added.
                step_count = sum(location[0] == name for location in inner_steps) - 1
                step_paths = tuple(f"{setting_path}/{index}" for index in range(step_count))
            else:
                step_paths = (setting_path,) if inner_steps[name,] is not None else ()
            settings.append(
                StepsSetting(
                    name=name,
                    path=setting_path,
                    title=field_info.title or name,
                    is_list=cardinality == "many",
                    minimum=property_schema.get("minItems", 0),
                    maximum=property_schema.get("maxItems"),
                    step_paths=step_paths,
                    item_label=hint_of(field_info).item_label or "check",
                )
            )
        return settings

    def label(self, path: str) -> str:
        """The step's name, the value of the answer it reads, or else its method's label."""
        step_data = self._step_data(path)
        if name := self._resolved_value(step_data.get("name")):
            return str(name)
        step = self._step(path)
        if isinstance(step.source, AnswerValueSource):
            return f"Value {step.source.value_number}"
        return step.method_label

    def notes(self, path: str) -> list[str]:
        """Notes beside the step's label: its question, input, weight and condition."""
        values = self.values(path)
        notes = [str(values["question"])] if values.get("question") else []
        if input_label := self._input_label(path):
            notes.append(f"Reads {input_label}")
        weight = values.get("weight", STEP_PROPERTIES["weight"].default)
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

    def _input_label(self, path: str) -> str | None:
        """What the step at ``path`` reads, in words, if it isn't the answer as it is."""
        match self._step(path).source:
            case AnswerValueSource(value_number=value_number):
                return f"value {value_number}"
            case OtherAnswerSource(question_id=question_id):
                return question_id
            case StepOutputSource(step_id=step_id):
                earlier_step_paths = self._earlier_step_paths(path)
                labels = _numbered(list(map(self._identifying_label, earlier_step_paths)))
                labels_by_id = dict(
                    zip(map(self._step_id, earlier_step_paths), labels, strict=True)
                )
                return f"the output of {labels_by_id.get(step_id, step_id)}"
        return None

    def describe(self, path: str) -> str:
        """What the step at ``path`` does; ``""`` until it is valid."""
        if not self.is_step(path):
            raise _not_a_step_error(path)
        place = self._snapshot.marking_places.get(path)
        step = place.step if place else None
        return "" if step is None or step.resolution.data_problems else step.describe()

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
        location = self._place(path).location
        return [
            json_pointer(sibling)
            for place in self._snapshot.places.values()
            if place.step is not None and is_within(location, place.location)
            for sibling in place.earlier_locations
        ]

    def _paths_by_id(self, paths: Iterable[str]) -> dict[str, str]:
        """The steps at ``paths`` that have an ID, by ID; the last of those sharing one."""
        return {step_id: path for path in paths if (step_id := self._step_id(path))}

    @property
    def step_problems(self) -> list[Problem]:
        """What is wrong with each step, the outermost included, at the step and field at
        fault."""
        return list(map(self._condition_order_problem, self._snapshot.problems))

    @property
    def problems(self) -> list[Problem]:
        """What stops it being finalized: its step problems, unless it is more than its steps."""
        return self.step_problems

    @property
    def _template_names(self) -> Mapping[str, Shape]:
        """What its templates may name besides earlier checks."""
        return {}

    def _resolved_value(self, value: Any) -> Any:
        """``value`` as the marking reads it."""
        return value

    def _condition_order_problem(self, problem: Problem) -> Problem:
        """``problem``, naming the check to move if its condition reads a later one."""
        condition = (
            value_at(self._data, f"{problem.step_path}/if") if problem.field == ("if",) else None
        )
        if not isinstance(condition, str):
            return problem
        location = location_of(problem.step_path)
        later_paths_by_id = {
            step_id: path
            for path, place in self._snapshot.places.items()
            if place.step is not None
            and any(is_within(location, sibling) for sibling in place.earlier_locations)
            and (step_id := self._step_id(path))
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

    def _step_data(self, path: str) -> JsonDict:
        """The step at ``path`` as YAML holds it."""
        if not self.is_step(path):
            raise _not_a_step_error(path)
        step_data: JsonDict = value_at(self._data, path)
        return step_data

    def _method(self, reference: str) -> Method:
        method = self._methods.by_ref.get(reference)
        if method is None:
            raise EditingError(f"There's no method {reference}.")
        return method

    def _step_id(self, path: str) -> str | None:
        step_id = value_at(self._data, path).get("id")
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


def _step_property(key: str, choices: Iterable[str] | None = None) -> JsonDict:
    """The JSON schema of a step's own property ``key``; with ``choices``, required to be one of
    them."""
    property_schema = _STEP_SCHEMA["properties"][key]
    if choices is None:
        return deepcopy(property_schema)
    return {
        "title": property_schema["title"],
        "description": property_schema["description"],
        **json_schema(literal_type(choices)),
    }


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
    """A step's settings and its own properties as one."""
    return {
        **settings_of(step_data),
        **{key: value for key, value in step_data.items() if key not in ("uses", "with")},
    }


def field_path(step_path: str, key: str) -> str:
    """The path of the field ``key`` of the step at ``step_path``."""
    return step_path + json_pointer((key,) if key in STEP_OWN_PROPERTIES else ("with", key))


def field_operation(step_path: str, step_data: Mapping[str, Any], key: str, value: Any) -> JsonDict:
    """The JSON Patch operation setting the field ``key`` of the step ``step_data`` at
    ``step_path``; a setting of a step without settings gives it them."""
    if key in STEP_OWN_PROPERTIES or isinstance(step_data.get("with"), Mapping):
        return {"op": "add", "path": field_path(step_path, key), "value": value}
    return {"op": "add", "path": f"{step_path}/with", "value": {key: value}}


def _is_available_at(method: Method, position: Position) -> bool:
    return (
        method.level != "internal"
        and method.can_run_at(position)
        and (method.gives_marks({}) or position.is_in_workflow)
    )


def first_unused_name(stem: str, taken: Container[object], separator: str = "_") -> str:
    """``stem``, or else the first of it numbered from 2 after ``separator`` that isn't taken."""
    numbered_names = (f"{stem}{separator}{number}" for number in count(2))
    return next(name for name in chain([stem], numbered_names) if name not in taken)


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _enclosing_step_path(location: Location) -> str:
    """The path of the step holding the step at ``location``, as ``/with/steps/0`` holds
    ``/with/steps/0/with/steps/1`` and ``/with/steps/0/with/check``."""
    is_in_list = bool(location) and isinstance(location[-1], int)
    return json_pointer(location[: -3 if is_in_list else -2])


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
