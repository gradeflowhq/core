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
from ..common.exceptions import EditingError, InvalidError, MethodFileError
from ..common.paths import get, join, parts, put
from ..common.problems import Problem, as_invalid, validation_problems
from ..common.schemas import JsonDict, json_schema
from ..common.text import quoted
from ..methods.catalogue import BUILTINS
from ..methods.check import CHECK_FIELDS, METHODS, Marking, check_of, holds_checks
from ..methods.collection import Methods, method_of
from ..methods.composed.file import (
    BODY,
    TEMPLATE_KEYS,
    Input,
    MethodFile,
    to_body,
    to_file_body,
)
from ..methods.example import Example, ExampleRow
from ..methods.method import Method
from ..methods.names import method_ref
from ..methods.settings import IDENTIFIER, annotated
from ..methods.templates import is_template, named_in, templated
from ..questions import new_question
from ..questions.question_set import QuestionSet
from ..questions.types import ALL_TYPES, QuestionId, QuestionType
from .drafts import MarkingDraft, field_path, fields_of, first_free
from .input_schemas import default_schema, input_schema
from .method_checks import (
    MethodChecksDraft,
    empty,
    fillable,
    input_reference,
)

METADATA = tuple(
    key for key in MethodFile.model_fields if key not in {"version", "inputs", "examples", *BODY}
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
    input: str | None
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
        if not isinstance(inputs, dict) or not all(isinstance(v, dict) for v in inputs.values()):
            raise EditingError("Inputs must be a mapping of names to their settings.")
        self.name, self.version = name, version
        self._methods = methods
        self._document = {key: deepcopy(data[key]) for key in data if key not in ("version", *BODY)}
        if "examples" in self._document:
            self._document["examples"] = _without_results(self._document["examples"])
        self._document["inputs"] = self._inputs = deepcopy(inputs)
        self.test_values = dict(test_values or {})
        question, body = _only(questions), to_body(data)
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
            raise InvalidError([Problem("A method definition must be a mapping.")])
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
        needed = marking.questions_read() | marking.questions_marked()
        questions = QuestionSet({key: questions[key] for key in questions if key in needed})
        check = _as_written(marking.dump())
        if _only(questions) is not None:
            check.pop("question", None)
        data = {"label": label, "summary": summary, **to_file_body(check)}
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
        """One question gives the method its answer type; more make it a group method."""
        if self.question:
            self._document.pop("group", None)
            if not self._document.get("question_types"):
                self._document["question_types"] = [self.questions[self.question].type]
        else:
            self._document.pop("question_types", None)
            self._document["group"] = bool(self.questions)

    def set_questions(self, questions: QuestionSet) -> None:
        """Change the questions, keeping the checks; a check of a removed question stays, as a
        problem to resolve."""
        previous, body = self.question, self._checks.data
        question = _only(questions)
        if previous and not question:
            for check in body["with"].get("checks", []):
                if isinstance(check, dict):
                    check.setdefault("question", previous)
            body.pop("question", None)
        if question:
            if previous and self._document.get("question_types") == [self.questions[previous].type]:
                self._document["question_types"] = [questions[question].type]
            for at in self._checks.paths():
                if get(body, at).get("question") == question:
                    get(body, at).pop("question")
            body["question"] = question
        self._checks = MethodChecksDraft(body, questions, question, self._methods, self._inputs)
        self._fit_to_questions()

    @property
    def label(self) -> str:
        """The method's name as people see it, as written so far."""
        return _text(self._document.get("label"))

    @property
    def summary(self) -> str:
        return _text(self._document.get("summary"))

    @property
    def question_types(self) -> frozenset[QuestionType]:
        """The questions it says it marks, as written so far."""
        written = self._document.get("question_types")
        if written == "all":
            return ALL_TYPES
        return frozenset(written) & ALL_TYPES if isinstance(written, list) else frozenset()

    @property
    def group(self) -> bool:
        """Whether it marks question groups."""
        return self._document.get("group") is True

    @property
    def metadata(self) -> dict[str, Any]:
        """What describes the method, as ``metadata_schema`` describes it: every answer type is
        listed rather than "all"."""
        data = {key: deepcopy(self._document[key]) for key in METADATA if key in self._document}
        if data.get("question_types") == "all":
            data["question_types"] = list(get_args(QuestionType))
        return data

    @property
    def metadata_schema(self) -> JsonDict:
        schema = json_schema(MethodFile)
        shown = [
            key
            for key in METADATA
            if key != "group" and (key, self.question) != ("question_types", None)
        ]
        schema["properties"] = {key: schema["properties"][key] for key in shown}
        schema["required"] = [key for key in schema.get("required", []) if key in shown]
        if kinds := schema["properties"].get("question_types"):
            # Chosen one by one; every one of them is "all".
            listed = next(member for member in kinds.pop("anyOf") if member["type"] == "array")
            schema["properties"]["question_types"] = {**kinds, **listed, "uniqueItems": True}
        return schema

    def update_metadata(self, **values: Any) -> None:
        """Change what describes the method. A value that is None, empty where optional or its
        default is left unset; every answer type is written as "all"."""
        if unknown := values.keys() - set(METADATA):
            raise EditingError(f"Unknown method metadata: {', '.join(sorted(unknown))}.")
        for key, value in values.items():
            field = MethodFile.model_fields[key]
            if value is None or value == field.default or (value == "" and not field.is_required()):
                self._document.pop(key, None)
            elif key == "question_types" and set(value) == ALL_TYPES:
                self._document[key] = "all"
            else:
                self._document[key] = deepcopy(value)
        self._fit_to_questions()

    @property
    def examples(self) -> tuple[Example, ...]:
        examples = self._document.get("examples", [])
        if not isinstance(examples, list):
            raise EditingError("Examples must be a list.")
        return tuple(Example.model_validate(example) for example in examples)

    def set_example(self, rows: Sequence[ExampleRow], *, at: int | None) -> int | None:
        """Make these rows the example at ``at`` with the questions and test values, adding one
        if ``at`` is None and removing it if there are no rows. Other examples stay. Where the
        example is now."""
        examples = list(self.examples)
        previous = examples[at] if at is not None else None
        if not rows:
            if at is not None:
                examples.pop(at)
            at = None
        else:
            example = Example(
                title=previous.title if previous else "Examples",
                description=previous.description if previous else "",
                questions=dict(self.questions),
                rows=list(rows),
                **{"with": deepcopy(self.test_values)},
            )
            if at is None:
                at = len(examples)
                examples.append(example)
            else:
                examples[at] = example
        self._document["examples"] = [_written(example) for example in examples]
        return at

    @property
    def inputs(self) -> dict[str, dict[str, Any]]:
        """Each input's declaration: its title, description, default and choices."""
        return deepcopy(self._inputs)

    @property
    def settings_schema(self) -> JsonDict:
        """The JSON schema of the inputs, as settings of a check of the method. Raises
        ``EditingError`` while the method can't be loaded."""
        try:
            return json_schema(self._load().settings)
        except MethodFileError as error:
            raise EditingError(str(error)) from None

    def input_schema(self, name: str) -> JsonDict:
        """The JSON schema of an input's declaration, as the setting it fills allows it."""
        field = next((field for field in self.fields() if field.input == name), None)
        default: JsonDict | None = None
        try:
            settings = self.settings_schema
            if (prop := settings.get("properties", {}).get(name)) is not None:
                default = {**prop, "$defs": settings.get("$defs", {})}
        except EditingError:
            if field is not None:
                spec = Input.model_validate(self._inputs[name])
                default = default_schema(spec, *self._field_type(field))
        return input_schema(default, self._schema_of(field) if field else None)

    def update_input(self, name: str, values: Mapping[str, Any]) -> None:
        """Replace an input's declaration; without a default, the input must be filled in. Raises
        ``InvalidError`` with a problem at each value the settings it fills don't take."""
        if not re.fullmatch(IDENTIFIER, name):
            raise EditingError(
                "Use letters, numbers and underscores; start with a letter or underscore."
            )
        if name in CHECK_FIELDS:
            raise EditingError(f"A check has its own {name}: name the input otherwise.")
        with as_invalid(values):
            value = Input.model_validate(values).model_dump(mode="json", exclude_unset=True)
        given = [
            *((("choices", n), choice) for n, choice in enumerate(value.get("choices") or [])),
            *((("examples", n), example) for n, example in enumerate(value.get("examples") or [])),
            *([(("default",), value["default"])] if "default" in value else []),
        ]
        found = [
            Problem(message, field=at)
            for field in self.fields()
            if field.input == name
            for at, given_value in given
            if (message := self._refusal(field, given_value))
        ]
        if "default" in value and value.get("choices") and value["default"] not in value["choices"]:
            found.append(Problem("Choose one of the input's choices.", field=("default",)))
        if found:
            raise InvalidError(list(dict.fromkeys(found)))
        self._checks.set_input(name, value)

    def remove_input(self, name: str) -> None:
        """Remove an input, fixing each field bound to it at its default."""
        for field in self.fields():
            if field.input == name:
                self.fix(field.path)
        self._checks.drop_input(name)
        self.test_values.pop(name, None)

    @property
    def _unused_inputs(self) -> set[str]:
        """Inputs that no check or template of the method names."""
        content = {
            "checks": self._checks.data,
            **{key: self._document.get(key) for key in ("description", "feedback")},
        }
        return self._inputs.keys() - named_in(content, "inputs")

    def fields(self) -> list[ConfigurableField]:
        """The settings an input can fill, in the order of their checks."""
        return [field for at in self._checks.paths() for field in self._fields(at)]

    def _fields(self, at: str) -> Iterator[ConfigurableField]:
        method = self._checks.method(at)
        if method is None:
            return
        node = get(self._checks.data, at)
        settings = method.settings_at(self._checks.position(at)).model_fields
        for key, prop in self._checks.schema(at)["properties"].items():
            if not fillable(key, settings) or (key not in settings and not at):
                continue
            bound = input_reference(get(node, field_path("", key)))
            yield ConfigurableField(
                path=field_path(at, key),
                check=at,
                key=key,
                title=prop.get("title", key),
                input=bound if bound in self._inputs else None,
            )

    def _schema_of(self, field: ConfigurableField) -> JsonDict:
        """The JSON schema of the setting, with the definitions it refers to."""
        schema = self._checks.schema(field.check)
        prop, defs = schema["properties"][field.key], schema.get("$defs", {})
        return deepcopy({**prop, "$defs": defs})

    def _field(self, path: str) -> ConfigurableField:
        """The setting at ``path`` an input can fill."""
        steps = parts(path)
        for end in range(len(steps), -1, -1):
            at = join(*steps[:end])
            if self._checks.is_check(at):
                break
        found = next((field for field in self._fields(at) if field.path == path), None)
        if found is None:
            raise EditingError(f"{path} isn't a setting an input can fill.")
        return found

    def expose(
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
        self.fix(path)
        field = self._field(path)
        given = default is not PydanticUndefined
        if not given:
            default = self._checks.values(field.check).get(
                field.key, self._schema_of(field).get("default")
            )
        if given or not empty(default):
            self._validate_default(field, default)
        stem = str(parts(path)[-1])
        name = first_free(f"{stem}_input" if stem in CHECK_FIELDS else stem, self._inputs)
        titles = {spec.get("title") for spec in self._inputs.values()}
        spec: dict[str, Any] = {"title": first_free(label or field.title, titles, " ")}
        if description:
            spec["description"] = description
        if given or not empty(default):
            spec["default"] = deepcopy(default)
        self._checks.set_input(name, spec)
        self._bind(path, name)
        return name

    def compatible_inputs(self, path: str) -> list[str]:
        """The inputs whose type and default can fill the setting at ``path``; an input no setting
        uses yet has no type."""
        field = self._field(path)
        types = {other.input: self._field_type(other)[0] for other in self.fields() if other.input}
        found = []
        for name, spec in self._inputs.items():
            if name in types and types[name] != self._field_type(field)[0]:
                continue
            try:
                if "default" in spec:
                    self._validate_default(field, spec["default"])
            except EditingError:
                continue
            found.append(name)
        return found

    def bind(self, path: str, name: str) -> None:
        """Fill a setting from an existing input, keeping the input's default."""
        field = self._field(path)
        if name not in self.compatible_inputs(path):
            raise EditingError(f"Input {name} can't fill {field.title}.")
        self._bind(path, name)
        self._drop_if_unused(field.input)

    def fix(self, path: str) -> None:
        """Give a bound setting its input's default as a fixed value."""
        field = self._field(path)
        if field.input is None:
            return
        data = self._checks.data
        put(data, path, deepcopy(self._inputs[field.input].get("default")))
        self._checks.replace(data)
        self._drop_if_unused(field.input)

    def _bind(self, path: str, name: str) -> None:
        data = self._checks.data
        put(data, path, f"{{{{ inputs.{name} }}}}")
        self._checks.replace(data)

    def _drop_if_unused(self, name: str | None) -> None:
        if name is not None and name in self._unused_inputs:
            self._checks.drop_input(name)
            self.test_values.pop(name, None)

    def _field_type(self, field: ConfigurableField) -> tuple[Any, FieldInfo]:
        method = self._checks.method(field.check)
        assert method is not None  # a configurable field belongs to a check of a known method
        settings = method.settings_at(self._checks.position(field.check)).model_fields
        info = settings.get(field.key) or CHECK_FIELDS[field.key]
        return info.annotation, info

    def _validate_default(self, field: ConfigurableField, value: Any) -> None:
        if message := self._refusal(field, value):
            raise EditingError(message)

    def _refusal(self, field: ConfigurableField, value: Any) -> str:
        """Why ``field`` doesn't take ``value``; ``""`` if it does."""
        annotation, info = self._field_type(field)
        try:
            TypeAdapter[Any](annotated(annotation, *info.metadata)).validate_python(
                value, context={METHODS: self._methods.by_ref}
            )
        except ValidationError as error:
            return "\n".join(map(str, validation_problems(error, value)))
        return ""

    @property
    def uneditable(self) -> list[str]:
        """What is written in ways that editing the checks field by field can't change."""
        data, paths = self._checks.data, self._checks.paths()
        found = (each for at in paths for each in self._uneditable(get(data, at), at, set(paths)))
        return list(dict.fromkeys(found))

    def _uneditable(self, node: dict[str, Any], at: str, checks: set[str]) -> Iterator[str]:
        """What of the check at ``at`` only YAML can change, given the paths of every check."""
        if "for_each" in node:
            yield "A check repeats for each item of an input."
            return
        method = self._checks.method(at)
        if method is None:
            yield f"There's no method {node['uses']}."
            return
        settings = node.get("with", {})
        if not isinstance(settings, dict):
            yield "A check's settings aren't a mapping."
            return
        known = method.settings.model_fields
        for key, value in fields_of(node).items():
            if key in TEMPLATE_KEYS:
                continue
            info = known.get(key) or CHECK_FIELDS.get(key)
            if info is None:
                yield f"{method.label} has no setting {key}."
            elif found := self._uneditable_field(info, value, field_path(at, key), checks):
                yield found

    def _uneditable_field(
        self, info: FieldInfo, value: Any, path: str, checks: set[str]
    ) -> str | None:
        held = holds_checks(info)
        if (held == "one" and path in checks) or (
            held == "many"
            and isinstance(value, list)
            and all(join(path, n) in checks for n in range(len(value)))
        ):
            return None
        if held is not None:
            return "A check list is an expression."
        if templated(value) and input_reference(value) not in self._inputs:
            return f"{info.title or path} is an expression."
        return None

    def to_yaml(self) -> str:
        """The method file, as it would be published."""
        return dump_yaml(self._file())

    def _file(self) -> dict[str, Any]:
        data = {"version": self.version, **deepcopy(self._document)}
        if not data["inputs"]:
            data.pop("inputs")
        body = self._checks.data
        body.pop("question", None)
        return data | to_file_body(body)

    @property
    def problems(self) -> list[Problem]:
        """What stops the method from being published or tried."""
        try:
            return self._problems(self._load())
        except MethodFileError as error:
            return [Problem(str(error))]

    def finalized(self) -> Method[Any]:
        """The method, once nothing is wrong with it. Raises ``EditingError`` with its first
        problem."""
        try:
            method = self._load()
        except MethodFileError as error:
            raise EditingError(str(error)) from None
        if found := self._problems(method):
            raise EditingError(str(found[0]))
        return method

    def _load(self) -> Method[Any]:
        others = Methods(method for method in self._methods if method.ref != self.ref).by_ref
        return method_of(self.name, self._file(), others)

    def _problems(self, method: Method[Any]) -> list[Problem]:
        if not self.questions:
            return [Problem("Add at least one question.")]
        if not method.label.strip():
            return [Problem("Give the method a name.")]
        if not method.group and not method.question_types:
            return [Problem("Choose at least one answer type.")]
        data: dict[str, Any] = {"uses": method.ref, "with": self.test_values}
        if self.question:
            data["question"] = self.question
        methods = Methods([*(item for item in self._methods if item.ref != method.ref), method])
        try:
            methods.dependencies(method.ref)
            check = check_of(data, methods.by_ref)
        except MethodFileError as error:
            return [Problem(str(error))]
        except ValidationError as error:
            found = validation_problems(error, data)
            return [_test_value_problem(problem, method, self.test_values) for problem in found]
        found = check.problems(self.questions)
        if found or not self.question or _holds_checks(method):
            return found
        others = sorted(method.question_types - {self.questions[self.question].type})
        return [
            Problem(f"It can't mark {kind} questions: {mismatch[0]}")
            for kind in others
            if (mismatch := check.problems(_sample(self.question, kind)))
        ]


def _test_value_problem(problem: Problem, method: Method[Any], given: Mapping[str, Any]) -> Problem:
    """A problem with the test values; one with a value not given asks for it by its input."""
    name = problem.field[0] if len(problem.field) == 1 and not problem.check else None
    info = method.settings.model_fields.get(str(name))
    if info is None or name in given:
        return problem
    return Problem(f"Give {quoted(info.title or str(name))} a test value or a default.")


def _without_results(examples: Any) -> Any:
    """Examples as written, without their rows' results, which only marking gives."""
    if not isinstance(examples, list):
        return examples
    return [
        {**example, "rows": [_without_result(row) for row in example["rows"]]}
        if isinstance(example, dict) and isinstance(example.get("rows"), list)
        else example
        for example in examples
    ]


def _without_result(row: Any) -> Any:
    return (
        {key: value for key, value in row.items() if key != "expected"}
        if isinstance(row, dict)
        else row
    )


def _written(example: Example) -> dict[str, Any]:
    """An example as YAML holds it, without defaults; its questions still say their type."""
    data = example.model_dump(mode="json", by_alias=True, exclude_defaults=True)
    data["questions"] = QuestionSet(example.questions).dump()
    return data


def _holds_checks(method: Method[Any]) -> bool:
    """Whether its settings take checks, which may suit fewer answer types than it does."""
    return any(holds_checks(info) is not None for info in method.settings.model_fields.values())


def _sample(key: QuestionId, kind: QuestionType) -> QuestionSet:
    """A new question of ``kind``, to try a method's checks on."""
    return QuestionSet({key: new_question(kind)})


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _only(questions: QuestionSet) -> QuestionId | None:
    return next(iter(questions)) if len(questions) == 1 else None


def _as_written(check: dict[str, Any]) -> dict[str, Any]:
    """A check whose answer text keeps its braces once it is in a method file's templates."""
    if "with" not in check:
        return check
    return {**check, "with": _raw(check["with"])}


def _raw(value: Any) -> Any:
    if isinstance(value, str) and is_template(value):
        return f"{{% raw %}}{value}{{% endraw %}}"
    if isinstance(value, dict):
        return (
            _as_written(value)
            if "uses" in value
            else {key: _raw(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_raw(item) for item in value]
    return value
