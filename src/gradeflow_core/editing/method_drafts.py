"""Writing a composed method: its steps are edited as a marking's are."""

import re
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, get_args

from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from ..common.documents import dump_yaml, load_yaml
from ..common.error_formatting import IDENTIFIER_RULE
from ..common.exceptions import EditingError
from ..common.paths import patched, value_at
from ..common.problems import InvalidError, Problem, invalid_errors, validation_problems
from ..common.schemas import JsonDict, json_schema
from ..common.text import quoted
from ..marking.examples import method_step_data
from ..methods.catalogue import BUILTINS
from ..methods.collection import Methods, method_of
from ..methods.composed.declared_settings import setting_name_problem
from ..methods.composed.file import (
    TEMPLATE_KEYS,
    WORKFLOW_KEYS,
    DeclaredSetting,
    MethodFile,
    as_file_steps,
    as_workflow,
)
from ..questions import new_question
from ..questions.question_set import QuestionSet
from ..questions.types import ALL_QUESTION_TYPES, QuestionId, QuestionType
from ..steps.example import Example, ExampleRow
from ..steps.method import Method
from ..steps.names import method_ref
from ..steps.places import marking_position
from ..steps.problems import located_problems
from ..steps.settings import IDENTIFIER, step_cardinality
from ..steps.step import METHODS_CONTEXT_KEY, STEP_OWN_PROPERTIES, Step, load_step
from ..steps.templates import has_template, is_template, names_read_in
from .drafts import MarkingDraft, field_operation, field_path, fields_of, first_unused_name
from .method_steps import (
    MethodStepsDraft,
    can_setting_fill,
    is_empty,
)
from .setting_schemas import declared_setting_schema, default_schema

METADATA_KEYS = tuple(
    key
    for key in MethodFile.model_fields
    if key not in {"version", "settings", "examples", *WORKFLOW_KEYS}
)
"""What describes a method, edited apart from its steps."""


@dataclass(frozen=True)
class ConfigurableField:
    """A field of a step that a declared setting of the method can fill."""

    step_path: str
    key: str
    title: str
    bound_setting: str | None

    @property
    def path(self) -> str:
        return field_path(self.step_path, self.key)


class MethodDraft:
    """A composed method being written for ``questions``, to become ``name`` at ``version``.

    One question makes a method for questions like it; more make a method for question groups.
    The steps are edited as ``steps_draft``; fields bound to declared settings read and change those
    settings' defaults. Test values fill declared settings when it is tried. Its examples hold no
    results unless ``set_example`` gives them.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        questions: QuestionSet,
        *,
        methods: Methods = BUILTINS,
        name: str,
        version: int,
        test_values: Mapping[str, Any] = {},
    ) -> None:
        declared_settings = data.get("settings", {})
        if not isinstance(declared_settings, dict) or not all(
            isinstance(declaration, dict) for declaration in declared_settings.values()
        ):
            raise EditingError("Settings must map each name to its declaration.")
        self.name, self.version = name, version
        self._methods = methods
        self._file_fields = {
            key: deepcopy(data[key]) for key in data if key not in ("version", *WORKFLOW_KEYS)
        }
        if "examples" in self._file_fields:
            self._file_fields["examples"] = _without_results(self._file_fields["examples"])
        self._file_fields["settings"] = self._declared_settings = deepcopy(declared_settings)
        question_id, workflow_data = _only_question_id(questions), as_workflow(data)
        if question_id:
            workflow_data["question"] = question_id
        self._steps_draft = MethodStepsDraft(
            workflow_data,
            questions,
            question_id,
            self._methods,
            self._declared_settings,
            test_values,
        )
        self._fit_to_questions()

    @classmethod
    def from_yaml(
        cls,
        text: str,
        questions: QuestionSet,
        *,
        methods: Methods = BUILTINS,
        name: str,
        version: int,
        test_values: Mapping[str, Any] = {},
    ) -> MethodDraft:
        """The method file ``text`` holds. Raises ``InvalidError`` for text that isn't YAML or a
        mapping."""
        data = load_yaml(text)
        if not isinstance(data, dict):
            raise InvalidError([Problem("A method file must be a set of keys and values.")])
        return cls(
            data, questions, methods=methods, name=name, version=version, test_values=test_values
        )

    @classmethod
    def new(
        cls,
        questions: QuestionSet,
        *,
        methods: Methods = BUILTINS,
        name: str,
        version: int,
        label: str,
        summary: str = "",
    ) -> MethodDraft:
        """A method called ``label`` with no steps yet."""
        data = {"label": label, "summary": summary, "steps": []}
        return cls(data, questions, methods=methods, name=name, version=version)

    @classmethod
    def from_marking(
        cls,
        marking: Step,
        questions: QuestionSet,
        *,
        methods: Methods = BUILTINS,
        name: str,
        version: int,
        label: str,
        summary: str = "",
    ) -> MethodDraft:
        """A method that marks as ``marking`` does, for the questions it reads; a workflow's
        steps become the method's own."""
        needed_questions = marking.questions_read() | marking.questions_marked()
        questions = QuestionSet(
            {
                question_id: questions[question_id]
                for question_id in questions
                if question_id in needed_questions
            }
        )
        check = _with_raw_templates(marking.dump())
        if _only_question_id(questions) is not None:
            check.pop("question", None)
        data = {"label": label, "summary": summary, **as_file_steps(check)}
        draft = cls(data, questions, methods=methods, name=name, version=version)
        draft.set_questions(questions)  # its steps needn't name the method's question
        return draft

    @property
    def test_values(self) -> dict[str, Any]:
        return self._steps_draft.test_values

    @test_values.setter
    def test_values(self, values: Mapping[str, Any]) -> None:
        self._steps_draft.test_values = values

    @property
    def questions(self) -> QuestionSet:
        return self._steps_draft.questions

    @property
    def question(self) -> QuestionId | None:
        """The only question, which makes it a method for questions like it."""
        return self._steps_draft.question

    @property
    def steps_draft(self) -> MarkingDraft:
        """The method's steps, edited as a marking of its questions."""
        return self._steps_draft

    def _fit_to_questions(self) -> None:
        """One question gives the method its question type; more make it a group method."""
        if self.question:
            self._file_fields.pop("marks_question_group", None)
            if not self._file_fields.get("question_types"):
                self._file_fields["question_types"] = [self.questions[self.question].type]
        else:
            self._file_fields.pop("question_types", None)
            self._file_fields["marks_question_group"] = bool(self.questions)

    def set_questions(self, questions: QuestionSet) -> None:
        """Change the questions, keeping the steps; a step of a removed question stays, as a
        problem."""
        previous_question_id, workflow_data = self.question, self._steps_draft.data
        question_id = _only_question_id(questions)
        if previous_question_id and not question_id:
            for check in workflow_data["with"].get("steps", []):
                if isinstance(check, dict):
                    check.setdefault("question", previous_question_id)
            workflow_data.pop("question", None)
        if question_id:
            if previous_question_id and self._file_fields.get("question_types") == [
                self.questions[previous_question_id].type
            ]:
                self._file_fields["question_types"] = [questions[question_id].type]
            for path in self._steps_draft.paths():
                if value_at(workflow_data, path).get("question") == question_id:
                    value_at(workflow_data, path).pop("question")
            workflow_data["question"] = question_id
        self._steps_draft = MethodStepsDraft(
            workflow_data,
            questions,
            question_id,
            self._methods,
            self._declared_settings,
            self.test_values,
        )
        self._fit_to_questions()

    @property
    def label(self) -> str:
        """The method's name as people see it, as written so far."""
        return _string_or_empty(self._file_fields.get("label"))

    @property
    def summary(self) -> str:
        return _string_or_empty(self._file_fields.get("summary"))

    @property
    def question_types(self) -> frozenset[QuestionType]:
        """The questions it says it marks, as written so far."""
        written_question_types = self._file_fields.get("question_types")
        if written_question_types == "all":
            return ALL_QUESTION_TYPES
        return (
            frozenset(written_question_types) & ALL_QUESTION_TYPES
            if isinstance(written_question_types, list)
            else frozenset()
        )

    @property
    def marks_question_group(self) -> bool:
        """Whether it marks question groups."""
        return self._file_fields.get("marks_question_group") is True

    @property
    def metadata(self) -> dict[str, Any]:
        """What describes the method, as ``metadata_schema`` describes it: every question type is
        listed rather than "all"."""
        data = {
            key: deepcopy(self._file_fields[key])
            for key in METADATA_KEYS
            if key in self._file_fields
        }
        if data.get("question_types") == "all":
            data["question_types"] = list(get_args(QuestionType))
        return data

    @property
    def metadata_schema(self) -> JsonDict:
        schema = json_schema(MethodFile)
        shown_keys = [
            key
            for key in METADATA_KEYS
            if key != "marks_question_group" and (key, self.question) != ("question_types", None)
        ]
        schema["properties"] = {key: schema["properties"][key] for key in shown_keys}
        schema["required"] = [key for key in schema.get("required", []) if key in shown_keys]
        if question_types_schema := schema["properties"].get("question_types"):
            # Chosen one by one; every one of them is "all".
            array_schema = next(
                member for member in question_types_schema.pop("anyOf") if member["type"] == "array"
            )
            schema["properties"]["question_types"] = {
                **question_types_schema,
                **array_schema,
                "uniqueItems": True,
            }
        return schema

    def update_metadata(self, **values: Any) -> None:
        """Change what describes the method. A value that is None, empty where optional or its
        default is left unset; every question type is written as "all"."""
        if unknown_keys := values.keys() - set(METADATA_KEYS):
            raise EditingError(f"Unknown method details: {', '.join(sorted(unknown_keys))}.")
        for key, value in values.items():
            field = MethodFile.model_fields[key]
            if value is None or value == field.default or (value == "" and not field.is_required()):
                self._file_fields.pop(key, None)
            elif key == "question_types" and set(value) == ALL_QUESTION_TYPES:
                self._file_fields[key] = "all"
            else:
                self._file_fields[key] = deepcopy(value)
        self._fit_to_questions()

    @property
    def examples(self) -> tuple[Example, ...]:
        examples = self._file_fields.get("examples", [])
        if not isinstance(examples, list):
            raise EditingError("Examples must be a list.")
        return tuple(Example.model_validate(example) for example in examples)

    def set_example(self, rows: Sequence[ExampleRow], *, index: int | None) -> int | None:
        """Make these rows the example at ``index`` with the questions and test values, adding
        one if ``index`` is None and removing it if there are no rows. Other examples stay. Where
        the example is now."""
        examples = list(self.examples)
        previous_example = examples[index] if index is not None else None
        if not rows:
            if index is not None:
                examples.pop(index)
            index = None
        else:
            example = Example(
                title=previous_example.title if previous_example else "Examples",
                description=previous_example.description if previous_example else "",
                questions=dict(self.questions),
                rows=list(rows),
                **{"with": deepcopy(self.test_values)},
            )
            if index is None:
                index = len(examples)
                examples.append(example)
            else:
                examples[index] = example
        self._file_fields["examples"] = [_example_data(example) for example in examples]
        return index

    @property
    def declared_settings(self) -> dict[str, dict[str, Any]]:
        """Each declared setting: its title, description, default and choices."""
        return deepcopy(self._declared_settings)

    @property
    def settings_schema(self) -> JsonDict:
        """The JSON schema of the declared settings, as settings of a step of the method. Raises
        ``EditingError`` while the method can't be loaded."""
        try:
            return json_schema(self._loaded_method().settings)
        except InvalidError as error:
            raise EditingError(str(error.problems[0])) from None

    def declared_setting_schema(self, name: str) -> JsonDict:
        """The JSON schema of a declared setting, as the setting it fills allows it."""
        field = next(
            (field for field in self.configurable_fields() if field.bound_setting == name), None
        )
        default_value_schema: JsonDict | None = None
        try:
            settings = self.settings_schema
            if (property_schema := settings.get("properties", {}).get(name)) is not None:
                default_value_schema = {**property_schema, "$defs": settings.get("$defs", {})}
        except EditingError:
            if field is not None:
                declaration = DeclaredSetting.model_validate(self._declared_settings[name])
                default_value_schema = default_schema(declaration, self._field_info(field))
        return declared_setting_schema(
            default_value_schema, self._schema_of(field) if field else None
        )

    def update_declared_setting(self, name: str, values: Mapping[str, Any]) -> None:
        """Replace a declared setting; without a default, it must be filled in. Raises
        ``InvalidError`` with a problem at each value the settings it fills don't take."""
        if not re.fullmatch(IDENTIFIER, name):
            raise EditingError(IDENTIFIER_RULE)
        if message := setting_name_problem(name):
            raise EditingError(message)
        with invalid_errors(values):
            declaration = DeclaredSetting.model_validate(values).model_dump(
                mode="json", exclude_unset=True
            )
        choices, examples = declaration.get("choices") or [], declaration.get("examples") or []
        values_to_check = [
            *((("choices", index), choice) for index, choice in enumerate(choices)),
            *((("examples", index), example) for index, example in enumerate(examples)),
            *([(("default",), declaration["default"])] if "default" in declaration else []),
        ]
        problems = [
            Problem(message, field=location)
            for field in self.configurable_fields()
            if field.bound_setting == name
            for location, value in values_to_check
            if (message := self._rejection_reason(field, value))
        ]
        if "default" in declaration and choices and declaration["default"] not in choices:
            problems.append(Problem("Choose one of the setting's choices.", field=("default",)))
        if problems:
            raise InvalidError(list(dict.fromkeys(problems)))
        self._steps_draft.set_declared_setting(name, declaration)

    def remove_declared_setting(self, name: str) -> None:
        """Remove a declared setting, fixing each field bound to it at its default."""
        for field in self.configurable_fields():
            if field.bound_setting == name:
                self.unbind(field.path)
        self._steps_draft.remove_declared_setting(name)

    def configurable_fields(self) -> list[ConfigurableField]:
        """The fields a declared setting can fill, in the order of their steps."""
        return [
            field for path in self._steps_draft.paths() for field in self._configurable_fields(path)
        ]

    def _configurable_fields(self, path: str) -> Iterator[ConfigurableField]:
        method = self._steps_draft.method(path)
        if method is None:
            return
        step_data = value_at(self._steps_draft.data, path)
        settings = method.settings_model_at(self._steps_draft.position(path)).model_fields
        for key, property_schema in self._steps_draft.schema(path)["properties"].items():
            if not can_setting_fill(key, settings) or (key not in settings and not path):
                continue
            yield ConfigurableField(
                step_path=path,
                key=key,
                title=property_schema.get("title", key),
                bound_setting=self._steps_draft.bound_setting(
                    value_at(step_data, field_path("", key))
                ),
            )

    def _schema_of(self, field: ConfigurableField) -> JsonDict:
        """The JSON schema of the setting, with the definitions it refers to."""
        schema = self._steps_draft.schema(field.step_path)
        property_schema, definitions = schema["properties"][field.key], schema.get("$defs", {})
        return deepcopy({**property_schema, "$defs": definitions})

    def _configurable_field(self, path: str) -> ConfigurableField:
        """The field at ``path`` a declared setting can fill."""
        field = next((field for field in self.configurable_fields() if field.path == path), None)
        if field is None:
            raise EditingError(f"{path} isn't a field a setting can fill.")
        return field

    def bind_new_setting(self, path: str) -> str:
        """Bind a field to a new declared setting that takes its title and value. The setting's
        name."""
        field = self._configurable_field(path)
        default = self._steps_draft.values(field.step_path).get(
            field.key, self._schema_of(field).get("default")
        )
        if not is_empty(default) and (message := self._rejection_reason(field, default)):
            raise EditingError(message)
        self.unbind(path)
        name = first_unused_name(
            f"{field.key}_setting" if field.key in STEP_OWN_PROPERTIES else field.key,
            self._declared_settings,
        )
        titles = {declaration.get("title") for declaration in self._declared_settings.values()}
        declaration: dict[str, Any] = {"title": first_unused_name(field.title, titles, " ")}
        if not is_empty(default):
            declaration["default"] = deepcopy(default)
        self._steps_draft.set_declared_setting(name, declaration)
        self._bind(field, name)
        return name

    def compatible_settings(self, path: str) -> list[str]:
        """The declared settings whose type and default can fill the field at ``path``."""
        field = self._configurable_field(path)
        annotations_by_setting_name = {
            bound_field.bound_setting: self._field_info(bound_field).annotation
            for bound_field in self.configurable_fields()
            if bound_field.bound_setting
        }
        annotation = self._field_info(field).annotation
        compatible_setting_names = []
        for name, declaration in self._declared_settings.items():
            if (
                name in annotations_by_setting_name
                and annotations_by_setting_name[name] != annotation
            ):
                continue
            if "default" in declaration and self._rejection_reason(field, declaration["default"]):
                continue
            compatible_setting_names.append(name)
        return compatible_setting_names

    def bind(self, path: str, name: str) -> None:
        """Fill a field from an existing declared setting, keeping its default."""
        field = self._configurable_field(path)
        if name not in self.compatible_settings(path):
            raise EditingError(f"Setting {name} can't fill {field.title}.")
        self._bind(field, name)
        if field.bound_setting is not None:
            self._remove_if_unused(field.bound_setting)

    def unbind(self, path: str) -> None:
        """Give a bound field its declared setting's default as a fixed value, or unset it without
        one."""
        field = self._configurable_field(path)
        if field.bound_setting is None:
            return
        declaration = self._declared_settings[field.bound_setting]
        operation = (
            {"op": "replace", "path": path, "value": deepcopy(declaration["default"])}
            if "default" in declaration
            else {"op": "remove", "path": path}
        )
        self._steps_draft.replace(patched(self._steps_draft.data, [operation]))
        self._remove_if_unused(field.bound_setting)

    def _bind(self, field: ConfigurableField, name: str) -> None:
        data = self._steps_draft.data
        template = f"{{{{ settings.{name} }}}}"
        step_data = value_at(data, field.step_path)
        operation = field_operation(field.step_path, step_data, field.key, template)
        self._steps_draft.replace(patched(data, [operation]))

    def _remove_if_unused(self, name: str) -> None:
        """Remove the declared setting ``name`` if no step or template of the method names it."""
        templates = {
            "steps": self._steps_draft.data,
            **{key: self._file_fields.get(key) for key in ("description", "feedback")},
        }
        if name not in names_read_in(templates, "settings"):
            self._steps_draft.remove_declared_setting(name)

    def _field_info(self, field: ConfigurableField) -> FieldInfo:
        method = self._steps_draft.method(field.step_path)
        assert method is not None  # a configurable field belongs to a step of a known method
        settings = method.settings_model_at(
            self._steps_draft.position(field.step_path)
        ).model_fields
        return settings.get(field.key) or STEP_OWN_PROPERTIES[field.key]

    def _rejection_reason(self, field: ConfigurableField, value: Any) -> str:
        """Why ``field`` doesn't take ``value``; ``""`` if it does."""
        try:
            TypeAdapter[Any](self._field_info(field).rebuild_annotation()).validate_python(
                value, context={METHODS_CONTEXT_KEY: self._methods.by_ref}
            )
        except ValidationError as error:
            return "\n".join(map(str, validation_problems(error, value)))
        return ""

    @property
    def uneditable_reasons(self) -> list[str]:
        """What is written in ways that editing the steps field by field can't change."""
        data, paths = self._steps_draft.data, self._steps_draft.paths()
        reasons = (
            reason
            for path in paths
            for reason in self._uneditable_reasons_of_step(value_at(data, path), path, set(paths))
        )
        return list(dict.fromkeys(reasons))

    def _uneditable_reasons_of_step(
        self, step_data: dict[str, Any], path: str, step_paths: set[str]
    ) -> Iterator[str]:
        """What of the step at ``path`` only YAML can change."""
        if "for_each" in step_data:
            yield "A step repeats for each item of a setting."
            return
        method = self._steps_draft.method(path)
        if method is None:
            yield f"There's no method {step_data['uses']}."
            return
        if not isinstance(step_data.get("with", {}), dict):
            yield "A step's settings aren't a mapping."
            return
        setting_fields = method.settings.model_fields
        for key, value in fields_of(step_data).items():
            if key in TEMPLATE_KEYS:
                continue
            field = setting_fields.get(key) or STEP_OWN_PROPERTIES.get(key)
            if field is None:
                yield f"{method.label} has no setting {key}."
            elif reason := self._uneditable_field_reason(
                field, value, field_path(path, key), step_paths
            ):
                yield reason

    def _uneditable_field_reason(
        self, field: FieldInfo, value: Any, path: str, step_paths: set[str]
    ) -> str | None:
        cardinality = step_cardinality(field)
        if (cardinality == "one" and path in step_paths) or (
            cardinality == "many"
            and isinstance(value, list)
            and all(f"{path}/{index}" in step_paths for index in range(len(value)))
        ):
            return None
        if cardinality is not None:
            return "A list of steps is written as a template."
        if has_template(value) and self._steps_draft.bound_setting(value) is None:
            return f"{field.title or path} is written as a template."
        return None

    def to_yaml(self) -> str:
        """The method file, as it would be published."""
        return dump_yaml(self._method_file())

    def _method_file(self) -> dict[str, Any]:
        data = {"version": self.version, **deepcopy(self._file_fields)}
        if not data["settings"]:
            data.pop("settings")
        workflow_data = self._steps_draft.data
        workflow_data.pop("question", None)
        return data | as_file_steps(workflow_data)

    @property
    def problems(self) -> list[Problem]:
        """What stops the method from being published or tried."""
        try:
            return self._problems(self._loaded_method())
        except InvalidError as error:
            return error.problems

    def finalized(self) -> Method:
        """The method, once nothing is wrong with it. Raises ``EditingError`` with its first
        problem."""
        try:
            method = self._loaded_method()
        except InvalidError as error:
            raise EditingError(str(error.problems[0])) from None
        if problems := self._problems(method):
            raise EditingError(str(problems[0]))
        return method

    def tested_step(self) -> Step:
        """The step a test run marks with: the method with the test values. Raises
        ``EditingError`` with the method's first problem."""
        return self._tested_step(self._methods_with(self.finalized()))

    def _other_methods(self) -> list[Method]:
        """The methods it may use: all but the one it replaces."""
        ref = method_ref(self.name, self.version)
        return [method for method in self._methods if method.ref != ref]

    def _loaded_method(self) -> Method:
        return method_of(self.name, self._method_file(), Methods(self._other_methods()).by_ref)

    def _methods_with(self, method: Method) -> Methods:
        return Methods([*self._other_methods(), method])

    def _tested_step(self, methods: Methods) -> Step:
        """The method with the test values, as a step using ``methods``, whatever is wrong with
        it."""
        test_step_data = method_step_data(
            method_ref(self.name, self.version), self.question, self.test_values
        )
        return load_step(test_step_data, methods.by_ref)

    def _problems(self, method: Method) -> list[Problem]:
        if not self.questions:
            return [Problem("Add at least one question.")]
        if not method.label.strip():
            return [Problem("Give the method a label.")]
        if not method.marks_question_group and not method.question_types:
            return [Problem("Choose at least one question type.")]
        if not self.uneditable_reasons and (problems := self._steps_draft.problems):
            return problems
        methods = self._methods_with(method)
        try:
            methods.dependencies(method.ref)
        except InvalidError as error:
            return error.problems
        step = self._tested_step(methods)
        if data_problems := step.resolution.data_problems:
            return [
                _test_value_problem(problem, method, self.test_values) for problem in data_problems
            ]
        problems = located_problems(step, marking_position(self.questions, step.question))
        if problems or not self.question or _holds_steps(method):
            return problems
        other_types = sorted(method.question_types - {self.questions[self.question].type})
        return [
            Problem(f"It can't mark {question_type} questions: {sample_problems[0]}")
            for question_type in other_types
            if (
                sample_problems := located_problems(
                    step,
                    marking_position(
                        QuestionSet({self.question: new_question(question_type)}), step.question
                    ),
                )
            )
        ]


def _test_value_problem(
    problem: Problem, method: Method, test_values: Mapping[str, Any]
) -> Problem:
    """A problem with the test values; one with a value not given asks for it by its setting."""
    name = problem.field[0] if len(problem.field) == 1 and not problem.step_path else None
    field = method.settings.model_fields.get(str(name))
    if field is None or name in test_values:
        return problem
    return Problem(f"Give {quoted(field.title or str(name))} a test value or a default.")


def _without_results(examples: Any) -> Any:
    """Examples as written, without their rows' results, which only marking gives."""
    if not isinstance(examples, list):
        return examples
    return [
        {
            **example,
            "rows": [
                {key: value for key, value in row.items() if key != "expected"}
                if isinstance(row, dict)
                else row
                for row in example["rows"]
            ],
        }
        if isinstance(example, dict) and isinstance(example.get("rows"), list)
        else example
        for example in examples
    ]


def _example_data(example: Example) -> dict[str, Any]:
    """An example as YAML holds it, without defaults; its questions still say their type."""
    data = example.model_dump(mode="json", by_alias=True, exclude_defaults=True)
    data["questions"] = QuestionSet(example.questions).dump()
    return data


def _holds_steps(method: Method) -> bool:
    """Whether its settings hold steps."""
    return any(
        step_cardinality(field) is not None for field in method.settings.model_fields.values()
    )


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _only_question_id(questions: QuestionSet) -> QuestionId | None:
    return next(iter(questions)) if len(questions) == 1 else None


def _with_raw_templates(value: Any) -> Any:
    """``value`` with its steps' templates in ``{% raw %}``, other than those under
    ``TEMPLATE_KEYS``."""
    if isinstance(value, str) and is_template(value):
        return f"{{% raw %}}{value}{{% endraw %}}"
    if isinstance(value, dict):
        return {
            key: item if key in TEMPLATE_KEYS and "uses" in value else _with_raw_templates(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_with_raw_templates(item) for item in value]
    return value
