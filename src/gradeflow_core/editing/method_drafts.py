"""Writing a composed method: its checks are edited as a marking's are."""

import re
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, get_args

from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ..common.documents import dump_yaml, load_yaml
from ..common.error_formatting import IDENTIFIER_RULE
from ..common.exceptions import EditingError, InvalidError, MethodFileError
from ..common.paths import get, join, path_parts, set_at
from ..common.problems import Problem, invalid_errors, validation_problems
from ..common.schemas import JsonDict, json_schema
from ..common.text import quoted
from ..methods.catalogue import BUILTINS
from ..methods.check import CHECK_FIELDS, METHODS_CONTEXT_KEY, Marking, check_of, how_many_checks
from ..methods.collection import Methods, method_of
from ..methods.composed.file import (
    COMBINATION_KEYS,
    TEMPLATE_KEYS,
    Input,
    MethodFile,
    as_combination,
    as_file_checks,
)
from ..methods.example import Example, ExampleRow
from ..methods.method import Method
from ..methods.names import method_ref
from ..methods.settings import IDENTIFIER, annotated
from ..methods.templates import has_template, is_template, names_read_in
from ..questions import new_question
from ..questions.question_set import QuestionSet
from ..questions.types import ALL_TYPES, QuestionId, QuestionType
from .drafts import MarkingDraft, field_path, fields_of, first_unused_name
from .input_schemas import default_schema, input_schema
from .method_checks import (
    MethodChecksDraft,
    can_input_fill,
    input_reference,
    is_empty,
)

METADATA_KEYS = tuple(
    key
    for key in MethodFile.model_fields
    if key not in {"version", "inputs", "examples", *COMBINATION_KEYS}
)
"""What describes a method, edited apart from its checks."""


@dataclass(frozen=True)
class ConfigurableField:
    """A setting of a check that an input of the method can fill."""

    path: str
    check: str
    """The path of the check it belongs to."""
    key: str
    title: str
    bound_input: str | None
    """The input it is bound to."""


class MethodDraft:
    """A composed method being written for ``questions``, to become ``name`` at ``version``.

    One question makes a method for questions like it; more make a method for question groups.
    The checks are edited as ``checks``; settings bound to inputs read and change the inputs'
    defaults. Test values fill inputs when it is tried, never changing their defaults. Its
    examples hold no results unless ``set_example`` gives them, such as from marking them.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        questions: QuestionSet,
        *,
        methods: Methods = BUILTINS,
        name: str,
        version: int,
        test_values: Mapping[str, Any] | None = None,
    ) -> None:
        inputs = data.get("inputs", {})
        if not isinstance(inputs, dict) or not all(
            isinstance(declaration, dict) for declaration in inputs.values()
        ):
            raise EditingError("Inputs must list names with their settings.")
        self.name, self.version = name, version
        self._methods = methods
        self._file_fields = {
            key: deepcopy(data[key]) for key in data if key not in ("version", *COMBINATION_KEYS)
        }
        if "examples" in self._file_fields:
            self._file_fields["examples"] = _without_results(self._file_fields["examples"])
        self._file_fields["inputs"] = self._inputs = deepcopy(inputs)
        self.test_values = dict(test_values or {})
        question, body = _only_question(questions), as_combination(data)
        if question:
            body["question"] = question
        self._checks = MethodChecksDraft(body, questions, question, self._methods, self._inputs)
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
        test_values: Mapping[str, Any] | None = None,
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
        """A method called ``label`` with no checks yet."""
        data = {"label": label, "summary": summary, "checks": []}
        return cls(data, questions, methods=methods, name=name, version=version)

    @classmethod
    def from_marking(
        cls,
        marking: Marking,
        questions: QuestionSet,
        *,
        methods: Methods = BUILTINS,
        name: str,
        version: int,
        label: str,
        summary: str = "",
    ) -> MethodDraft:
        """A method that marks as ``marking`` does, for the questions it reads; a Combination's
        checks become the method's own."""
        needed_questions = marking.questions_read() | marking.questions_marked()
        questions = QuestionSet(
            {key: questions[key] for key in questions if key in needed_questions}
        )
        check = _with_raw_templates(marking.dump())
        if _only_question(questions) is not None:
            check.pop("question", None)
        data = {"label": label, "summary": summary, **as_file_checks(check)}
        draft = cls(data, questions, methods=methods, name=name, version=version)
        draft.set_questions(questions)  # its checks needn't name the method's question
        return draft

    @property
    def questions(self) -> QuestionSet:
        return self._checks.questions

    @property
    def question(self) -> QuestionId | None:
        """The only question, which makes it a method for questions like it."""
        return self._checks.question

    @property
    def checks(self) -> MarkingDraft:
        """The method's checks, edited as the marking of its questions."""
        return self._checks

    @property
    def ref(self) -> str:
        return method_ref(self.name, self.version)

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
        """Change the questions, keeping the checks; a check of a removed question stays, as a
        problem to resolve."""
        previous, body = self.question, self._checks.data
        question = _only_question(questions)
        if previous and not question:
            for check in body["with"].get("checks", []):
                if isinstance(check, dict):
                    check.setdefault("question", previous)
            body.pop("question", None)
        if question:
            if previous and self._file_fields.get("question_types") == [
                self.questions[previous].type
            ]:
                self._file_fields["question_types"] = [questions[question].type]
            for path in self._checks.paths():
                if get(body, path).get("question") == question:
                    get(body, path).pop("question")
            body["question"] = question
        self._checks = MethodChecksDraft(body, questions, question, self._methods, self._inputs)
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
        written = self._file_fields.get("question_types")
        if written == "all":
            return ALL_TYPES
        return frozenset(written) & ALL_TYPES if isinstance(written, list) else frozenset()

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
        if unknown := values.keys() - set(METADATA_KEYS):
            raise EditingError(f"Unknown method details: {', '.join(sorted(unknown))}.")
        for key, value in values.items():
            field = MethodFile.model_fields[key]
            if value is None or value == field.default or (value == "" and not field.is_required()):
                self._file_fields.pop(key, None)
            elif key == "question_types" and set(value) == ALL_TYPES:
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
        previous = examples[index] if index is not None else None
        if not rows:
            if index is not None:
                examples.pop(index)
            index = None
        else:
            example = Example(
                title=previous.title if previous else "Examples",
                description=previous.description if previous else "",
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
    def inputs(self) -> dict[str, dict[str, Any]]:
        """Each input's declaration: its title, description, default and choices."""
        return deepcopy(self._inputs)

    @property
    def settings_schema(self) -> JsonDict:
        """The JSON schema of the inputs, as settings of a check of the method. Raises
        ``EditingError`` while the method can't be loaded."""
        try:
            return json_schema(self._loaded_method().settings)
        except MethodFileError as error:
            raise EditingError(str(error)) from None

    def input_schema(self, name: str) -> JsonDict:
        """The JSON schema of an input's declaration, as the setting it fills allows it."""
        field = next(
            (field for field in self.configurable_fields() if field.bound_input == name), None
        )
        default: JsonDict | None = None
        try:
            settings = self.settings_schema
            if (property_schema := settings.get("properties", {}).get(name)) is not None:
                default = {**property_schema, "$defs": settings.get("$defs", {})}
        except EditingError:
            if field is not None:
                declaration = Input.model_validate(self._inputs[name])
                default = default_schema(declaration, *self._field_annotation(field))
        return input_schema(default, self._schema_of(field) if field else None)

    def update_input(self, name: str, values: Mapping[str, Any]) -> None:
        """Replace an input's declaration; without a default, the input must be filled in. Raises
        ``InvalidError`` with a problem at each value the settings it fills don't take."""
        if not re.fullmatch(IDENTIFIER, name):
            raise EditingError(IDENTIFIER_RULE)
        if name in CHECK_FIELDS:
            raise EditingError(f"A check has its own {name}: name the input otherwise.")
        with invalid_errors(values):
            declaration = Input.model_validate(values).model_dump(mode="json", exclude_unset=True)
        choices, examples = declaration.get("choices") or [], declaration.get("examples") or []
        values_to_check = [
            *((("choices", index), choice) for index, choice in enumerate(choices)),
            *((("examples", index), example) for index, example in enumerate(examples)),
            *([(("default",), declaration["default"])] if "default" in declaration else []),
        ]
        problems = [
            Problem(message, field=location)
            for field in self.configurable_fields()
            if field.bound_input == name
            for location, value in values_to_check
            if (message := self._rejection_reason(field, value))
        ]
        if "default" in declaration and choices and declaration["default"] not in choices:
            problems.append(Problem("Choose one of the input's choices.", field=("default",)))
        if problems:
            raise InvalidError(list(dict.fromkeys(problems)))
        self._checks.set_input(name, declaration)

    def remove_input(self, name: str) -> None:
        """Remove an input, fixing each field bound to it at its default."""
        for field in self.configurable_fields():
            if field.bound_input == name:
                self.unbind(field.path)
        self._checks.remove_input(name)
        self.test_values.pop(name, None)

    @property
    def _unused_inputs(self) -> set[str]:
        """Inputs that no check or template of the method names."""
        templates = {
            "checks": self._checks.data,
            **{key: self._file_fields.get(key) for key in ("description", "feedback")},
        }
        return self._inputs.keys() - names_read_in(templates, "inputs")

    def configurable_fields(self) -> list[ConfigurableField]:
        """The settings an input can fill, in the order of their checks."""
        return [field for path in self._checks.paths() for field in self._configurable_fields(path)]

    def _configurable_fields(self, path: str) -> Iterator[ConfigurableField]:
        method = self._checks.method(path)
        if method is None:
            return
        node = get(self._checks.data, path)
        settings = method.settings_model_at(self._checks.position(path)).model_fields
        for key, property_schema in self._checks.schema(path)["properties"].items():
            if not can_input_fill(key, settings) or (key not in settings and not path):
                continue
            input_name = input_reference(get(node, field_path("", key)))
            yield ConfigurableField(
                path=field_path(path, key),
                check=path,
                key=key,
                title=property_schema.get("title", key),
                bound_input=input_name if input_name in self._inputs else None,
            )

    def _schema_of(self, field: ConfigurableField) -> JsonDict:
        """The JSON schema of the setting, with the definitions it refers to."""
        schema = self._checks.schema(field.check)
        property_schema, definitions = schema["properties"][field.key], schema.get("$defs", {})
        return deepcopy({**property_schema, "$defs": definitions})

    def _configurable_field(self, path: str) -> ConfigurableField:
        """The setting at ``path`` an input can fill."""
        segments = path_parts(path)
        for end in range(len(segments), -1, -1):
            check_path = join(*segments[:end])
            if self._checks.is_check(check_path):
                break
        field = next(
            (field for field in self._configurable_fields(check_path) if field.path == path), None
        )
        if field is None:
            raise EditingError(f"{path} isn't a setting an input can fill.")
        return field

    def bind_new_input(
        self,
        path: str,
        *,
        label: str | None = None,
        description: str = "",
        default: Any = PydanticUndefined,
    ) -> str:
        """Bind a setting to a new input of its own, labelled as the setting unless given, whose
        default is the setting's value unless given, numbered if another input has its label. The
        input's name."""
        self.unbind(path)
        field = self._configurable_field(path)
        has_default = default is not PydanticUndefined
        if not has_default:
            default = self._checks.values(field.check).get(
                field.key, self._schema_of(field).get("default")
            )
        if has_default or not is_empty(default):
            self._validate_default(field, default)
        stem = str(path_parts(path)[-1])
        name = first_unused_name(f"{stem}_input" if stem in CHECK_FIELDS else stem, self._inputs)
        titles = {declaration.get("title") for declaration in self._inputs.values()}
        declaration: dict[str, Any] = {
            "title": first_unused_name(label or field.title, titles, " ")
        }
        if description:
            declaration["description"] = description
        if has_default or not is_empty(default):
            declaration["default"] = deepcopy(default)
        self._checks.set_input(name, declaration)
        self._bind(path, name)
        return name

    def compatible_inputs(self, path: str) -> list[str]:
        """The inputs whose type and default can fill the setting at ``path``; an input no setting
        uses yet has no type."""
        field = self._configurable_field(path)
        types = {
            other.bound_input: self._field_annotation(other)[0]
            for other in self.configurable_fields()
            if other.bound_input
        }
        compatible = []
        for name, declaration in self._inputs.items():
            if name in types and types[name] != self._field_annotation(field)[0]:
                continue
            try:
                if "default" in declaration:
                    self._validate_default(field, declaration["default"])
            except EditingError:
                continue
            compatible.append(name)
        return compatible

    def bind(self, path: str, name: str) -> None:
        """Fill a setting from an existing input, keeping the input's default."""
        field = self._configurable_field(path)
        if name not in self.compatible_inputs(path):
            raise EditingError(f"Input {name} can't fill {field.title}.")
        self._bind(path, name)
        self._remove_if_unused(field.bound_input)

    def unbind(self, path: str) -> None:
        """Give a bound setting its input's default as a fixed value."""
        field = self._configurable_field(path)
        if field.bound_input is None:
            return
        data = self._checks.data
        set_at(data, path, deepcopy(self._inputs[field.bound_input].get("default")))
        self._checks.replace(data)
        self._remove_if_unused(field.bound_input)

    def _bind(self, path: str, name: str) -> None:
        data = self._checks.data
        set_at(data, path, f"{{{{ inputs.{name} }}}}")
        self._checks.replace(data)

    def _remove_if_unused(self, name: str | None) -> None:
        if name is not None and name in self._unused_inputs:
            self._checks.remove_input(name)
            self.test_values.pop(name, None)

    def _field_annotation(self, field: ConfigurableField) -> tuple[Any, FieldInfo]:
        method = self._checks.method(field.check)
        assert method is not None  # a configurable field belongs to a check of a known method
        settings = method.settings_model_at(self._checks.position(field.check)).model_fields
        field_info = settings.get(field.key) or CHECK_FIELDS[field.key]
        return field_info.annotation, field_info

    def _validate_default(self, field: ConfigurableField, value: Any) -> None:
        if message := self._rejection_reason(field, value):
            raise EditingError(message)

    def _rejection_reason(self, field: ConfigurableField, value: Any) -> str:
        """Why ``field`` doesn't take ``value``; ``""`` if it does."""
        annotation, field_info = self._field_annotation(field)
        try:
            TypeAdapter[Any](annotated(annotation, *field_info.metadata)).validate_python(
                value, context={METHODS_CONTEXT_KEY: self._methods.by_ref}
            )
        except ValidationError as error:
            return "\n".join(map(str, validation_problems(error, value)))
        return ""

    @property
    def uneditable_reasons(self) -> list[str]:
        """What is written in ways that editing the checks field by field can't change."""
        data, paths = self._checks.data, self._checks.paths()
        reasons = (
            reason
            for path in paths
            for reason in self._uneditable(get(data, path), path, set(paths))
        )
        return list(dict.fromkeys(reasons))

    def _uneditable(self, node: dict[str, Any], path: str, checks: set[str]) -> Iterator[str]:
        """What of the check at ``path`` only YAML can change, given the paths of every check."""
        if "for_each" in node:
            yield "A check repeats for each item of an input."
            return
        method = self._checks.method(path)
        if method is None:
            yield f"There's no method {node['uses']}."
            return
        settings = node.get("with", {})
        if not isinstance(settings, dict):
            yield "A check's settings aren't a mapping."
            return
        setting_fields = method.settings.model_fields
        for key, value in fields_of(node).items():
            if key in TEMPLATE_KEYS:
                continue
            field = setting_fields.get(key) or CHECK_FIELDS.get(key)
            if field is None:
                yield f"{method.label} has no setting {key}."
            elif reason := self._uneditable_field(field, value, field_path(path, key), checks):
                yield reason

    def _uneditable_field(
        self, field: FieldInfo, value: Any, path: str, checks: set[str]
    ) -> str | None:
        holds = how_many_checks(field)
        if (holds == "one" and path in checks) or (
            holds == "many"
            and isinstance(value, list)
            and all(join(path, index) in checks for index in range(len(value)))
        ):
            return None
        if holds is not None:
            return "A check list is written as a template."
        if has_template(value) and input_reference(value) not in self._inputs:
            return f"{field.title or path} is written as a template."
        return None

    def to_yaml(self) -> str:
        """The method file, as it would be published."""
        return dump_yaml(self._method_file())

    def _method_file(self) -> dict[str, Any]:
        data = {"version": self.version, **deepcopy(self._file_fields)}
        if not data["inputs"]:
            data.pop("inputs")
        body = self._checks.data
        body.pop("question", None)
        return data | as_file_checks(body)

    @property
    def problems(self) -> list[Problem]:
        """What stops the method from being published or tried."""
        try:
            return self._problems(self._loaded_method())
        except MethodFileError as error:
            return [Problem(str(error))]

    def finalized(self) -> Method[Any]:
        """The method, once nothing is wrong with it. Raises ``EditingError`` with its first
        problem."""
        try:
            method = self._loaded_method()
        except MethodFileError as error:
            raise EditingError(str(error)) from None
        if problems := self._problems(method):
            raise EditingError(str(problems[0]))
        return method

    def _loaded_method(self) -> Method[Any]:
        other_methods = Methods(method for method in self._methods if method.ref != self.ref).by_ref
        return method_of(self.name, self._method_file(), other_methods)

    def _problems(self, method: Method[Any]) -> list[Problem]:
        if not self.questions:
            return [Problem("Add at least one question.")]
        if not method.label.strip():
            return [Problem("Give the method a label.")]
        if not method.marks_question_group and not method.question_types:
            return [Problem("Choose at least one question type.")]
        test_check: dict[str, Any] = {"uses": method.ref, "with": self.test_values}
        if self.question:
            test_check["question"] = self.question
        methods = Methods([*(other for other in self._methods if other.ref != method.ref), method])
        try:
            methods.dependencies(method.ref)
            check = check_of(test_check, methods.by_ref)
        except MethodFileError as error:
            return [Problem(str(error))]
        except ValidationError as error:
            problems = validation_problems(error, test_check)
            return [_test_value_problem(problem, method, self.test_values) for problem in problems]
        problems = check.problems(self.questions)
        if problems or not self.question or _takes_checks(method):
            return problems
        other_types = sorted(method.question_types - {self.questions[self.question].type})
        return [
            Problem(f"It can't mark {question_type} questions: {sample_problems[0]}")
            for question_type in other_types
            if (sample_problems := check.problems(_sample_questions(self.question, question_type)))
        ]


def _test_value_problem(
    problem: Problem, method: Method[Any], test_values: Mapping[str, Any]
) -> Problem:
    """A problem with the test values; one with a value not given asks for it by its input."""
    name = problem.field[0] if len(problem.field) == 1 and not problem.check else None
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


def _takes_checks(method: Method[Any]) -> bool:
    """Whether its settings take checks, which may suit fewer question types than it does."""
    return any(
        how_many_checks(field) is not None for field in method.settings.model_fields.values()
    )


def _sample_questions(key: QuestionId, question_type: QuestionType) -> QuestionSet:
    """A new question of ``question_type``, to try a method's checks on."""
    return QuestionSet({key: new_question(question_type)})


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _only_question(questions: QuestionSet) -> QuestionId | None:
    return next(iter(questions)) if len(questions) == 1 else None


def _with_raw_templates(value: Any) -> Any:
    """``value`` with the answer text of its checks keeping its braces once it is in a method
    file's templates."""
    if isinstance(value, str) and is_template(value):
        return f"{{% raw %}}{value}{{% endraw %}}"
    if isinstance(value, dict) and "uses" in value:
        return {**value, "with": _with_raw_templates(value["with"])} if "with" in value else value
    if isinstance(value, dict):
        return {key: _with_raw_templates(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_raw_templates(item) for item in value]
    return value
