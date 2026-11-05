"""A composed method's inputs, each typed by the setting it fills."""

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, get_args, get_origin

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ...common.error_formatting import error_message
from ...common.hints import Hint, hint_of
from ...common.paths import Location
from ...common.text import quoted_join
from ..check import CHECK_FIELDS, METHODS_CONTEXT_KEY, how_many_checks
from ..settings import Settings, annotated, item_field, literal
from ..templates import TemplateError, has_template, names_read_in, reference, validate
from .expand import EXPANDED_NAMES
from .file import COMBINATION, TEMPLATE_KEYS, Input, MethodFile, MethodFileFieldError

if TYPE_CHECKING:
    from ..method import Method

_DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")
"""A reference by name, such as ``inputs.points``."""


class _InputTypes:
    """Each input's type and field: those of the setting it fills, where the checks name it."""

    def __init__(self, file: MethodFile, methods_by_ref: Mapping[str, Method[Any]]) -> None:
        self.file = file
        self.methods_by_ref = methods_by_ref
        self.input_types: dict[str, tuple[Any, FieldInfo]] = {}
        self.repeated_item_fields: list[dict[str, tuple[Any, FieldInfo]]] = []
        self.inputs_in_templates = names_read_in([file.description, file.feedback], "inputs")
        """Inputs that templates and conditions name."""
        self.errors: list[MethodFileFieldError] = []
        """The problems found so far."""

    def inputs(self) -> dict[str, tuple[Any, FieldInfo]]:
        combination = self.methods_by_ref.get(COMBINATION)
        if combination is None:
            raise MethodFileFieldError(f"There's no method {COMBINATION}.", ("uses",))
        self._visit_settings(self.file.combination()["with"], combination, ())
        for key, declared_input in self.file.inputs.items():
            with self._collect_errors():
                self._visit_input(key, declared_input)
        if self.errors:
            raise MethodFileFieldError.combined(self.errors)
        return self.input_types

    def _visit_input(self, key: str, declared_input: Input) -> None:
        if key in CHECK_FIELDS:
            raise MethodFileFieldError(
                f"A check has its own {key}: name the input otherwise.", ("inputs", key)
            )
        if (
            key in self.input_types or self.errors
        ):  # a setting that failed may be the one that uses it
            return
        if key not in self.inputs_in_templates:
            raise MethodFileFieldError(
                f"{declared_input.title or f'Input {key}'} isn't used.", ("inputs", key)
            )
        self.input_types[key] = (_type_of(declared_input.default), FieldInfo())

    @contextmanager
    def _collect_errors(self) -> Iterator[None]:
        """Records a problem raised inside, and goes on."""
        try:
            yield
        except MethodFileFieldError as error:
            self.errors.append(error)

    def _record(self, expression: str, annotation: Any, info: FieldInfo, path: Location) -> None:
        if not _DOTTED_NAME.fullmatch(expression):
            # A template's own refusal says best what's wrong, such as arithmetic.
            try:
                validate(
                    expression, dict.fromkeys(EXPANDED_NAMES, "any"), expression_kind="reference"
                )
            except TemplateError as error:
                raise MethodFileFieldError(str(error), path) from None
            raise MethodFileFieldError(
                "A setting takes an input as it is, such as {{ inputs.name }}.", path
            )
        root, *attribute_names = expression.split(".")
        if (
            root == "inputs"
            and len(attribute_names) == 1
            and attribute_names[0] in self.file.inputs
        ):
            existing = self.input_types.get(attribute_names[0])
            if existing is not None and existing[0] != annotation:
                raise MethodFileFieldError(
                    f"Input {attribute_names[0]} fills settings of different types.", path
                )
            self.input_types.setdefault(attribute_names[0], (annotation, info))
        elif root == "item" and self.repeated_item_fields and len(attribute_names) <= 1:
            self.repeated_item_fields[-1].setdefault(
                attribute_names[0] if attribute_names else "", (annotation, info)
            )
        elif root != "index":
            raise MethodFileFieldError(f"There's no {expression}.", path)

    def _visit_check(self, node: Any, path: Location) -> None:
        if not isinstance(node, Mapping) or "uses" not in node:
            raise MethodFileFieldError("A check needs uses: and a method's name.", path)
        method = self.methods_by_ref.get(str(node["uses"]))
        if method is None:
            raise MethodFileFieldError(f"There's no method {node['uses']}.", (*path, "uses"))
        for key, value in sorted(node.items(), key=lambda item: item[0] != "with"):
            if key in TEMPLATE_KEYS:
                self.inputs_in_templates |= names_read_in(value, "inputs")
                continue
            if key in ("uses", "for_each"):
                continue
            if key == "with":
                self._visit_settings(value, method, (*path, "with"))
            elif key not in CHECK_FIELDS:
                raise MethodFileFieldError(f"A check has no {key}:.", (*path, key))
            elif (expression := reference(value)) is not None:
                info = CHECK_FIELDS[key]
                self._record(expression, info.annotation, info, (*path, key))

    def _visit_settings(self, values: Any, method: Method[Any], path: Location) -> None:
        if (expression := reference(values)) is not None:
            raise MethodFileFieldError(
                f"Fill in a method's settings one by one, not with {expression}.",
                path,
            )
        if not isinstance(values, Mapping | None):
            raise MethodFileFieldError("A check's settings aren't a mapping.", path)
        model = method.settings
        for key, value in (values or {}).items():
            with self._collect_errors():
                info = model.model_fields.get(key)
                if info is None:
                    raise MethodFileFieldError(
                        f"{method.label} has no setting {key}.", (*path, key)
                    )
                if (source := _source_of(value)) is not None:
                    info = _with_input_conditions(info, source, values or {}, model, (*path, key))
                self._visit_setting(value, info, (*path, key))

    def _visit_setting(self, value: Any, info: FieldInfo, path: Location) -> None:
        if (expression := reference(value)) is not None:
            self._record(expression, info.annotation, info, path)
            return
        match how_many_checks(info):
            case "one":
                self._visit_check(value, path)
            case "many" if (item_info := item_field(info)) is not None:
                for index, item in enumerate(value):
                    with self._collect_errors():
                        self._visit_item(item, item_info, (*path, index))
            case _ if has_template(value):
                self.inputs_in_templates |= names_read_in(value, "inputs")
                element = get_args(info.annotation)[0] if get_args(info.annotation) else Any
                for index, item in enumerate(value if isinstance(value, list) else ()):
                    if (expression := reference(item)) is not None:
                        self._record(expression, element, info, (*path, index))
                    elif not has_template(item) and (item_info := item_field(info)):
                        _validate_literal(item, item_info, (*path, index), self.methods_by_ref)
            case _:
                _validate_literal(value, info, path, self.methods_by_ref)

    def _visit_item(self, item: Any, info: FieldInfo, path: Location) -> None:
        if (expression := reference(item)) is not None:
            self._record(expression, annotated(info.annotation, *info.metadata), info, path)
            return
        if isinstance(item, Mapping) and "for_each" in item:
            source = reference(item["for_each"])
            if source is None or not source.startswith("inputs."):
                raise MethodFileFieldError(
                    "for_each needs one {{ inputs.… }}.", (*path, "for_each")
                )
            self.repeated_item_fields.append({})
            self._visit_check(item, path)
            item_fields = self._with_item_titles(self.repeated_item_fields.pop(), source, path)
            if "" in item_fields:
                annotation, field = item_fields[""]
                annotation = annotated(annotation, *field.metadata)
            elif item_fields:
                annotation = create_model(
                    "Item",
                    **{
                        key: (annotation, field) for key, (annotation, field) in item_fields.items()
                    },
                )  # type: ignore[call-overload]
                field = FieldInfo()
            else:
                raise MethodFileFieldError("A repeated check uses {{ item }}.", path)
            self._record(
                source,
                list[annotation],  # type: ignore[valid-type]
                FieldInfo(title=field.title, json_schema_extra=field.json_schema_extra),
                (*path, "for_each"),
            )
            return
        self._visit_check(item, path)

    def _with_item_titles(
        self, item_fields: dict[str, tuple[Any, FieldInfo]], source: str, path: Location
    ) -> dict[str, tuple[Any, FieldInfo]]:
        """The fields of each item, with the titles the input gives them."""
        declared_input = self.file.inputs.get(source.removeprefix("inputs."))
        item_titles = (declared_input.items if declared_input else None) or {}
        if unused := item_titles.keys() - item_fields.keys():
            raise MethodFileFieldError(
                f"Each item has no {', '.join(sorted(unused))}.", (*path, "for_each")
            )
        return {
            key: (
                annotation,
                _merged(field, Field(**item_titles[key].model_dump(exclude_none=True))),
            )
            if key in item_titles
            else (annotation, field)
            for key, (annotation, field) in item_fields.items()
        }


def _source_of(value: Any) -> str | None:
    """What fills a setting from elsewhere: ``inputs``, or ``item`` of a repeated check."""
    expression = reference(value)
    if expression is None and isinstance(value, list):
        expression = next(
            (item_expression for item in value if (item_expression := reference(item))), None
        )
    return expression.split(".")[0] if expression else None


def _with_input_conditions(
    info: FieldInfo,
    source: str,
    siblings: Mapping[str, Any],
    model: type[BaseModel],
    path: Location,
) -> FieldInfo:
    """``info`` filled from ``source``, applying when what fills the settings it depends on
    does: a condition on a setting filled from the same place names what fills it, and one on a
    fixed setting holds or refuses the file."""
    hint = hint_of(info)
    conditions: dict[str, JsonValue] = {}
    for key, values in (hint.applies_when or {}).items():
        sibling = model.model_fields[key]
        value = siblings.get(key, sibling.get_default(call_default_factory=True))
        options = values if isinstance(values, list) else [values]
        if (expression := reference(value)) is not None:
            scope, _, name = expression.partition(".")
            if scope == source and name:
                conditions[name] = values
        elif not has_template(value) and getattr(value, "value", value) not in options:
            labels = hint_of(sibling).labels or {}
            allowed_values = quoted_join(
                [labels.get(str(option), option) for option in options], conjunction="or"
            )
            raise MethodFileFieldError(
                f"{info.title} applies when {sibling.title} is {allowed_values}.", path
            )
    if conditions == (hint.applies_when or {}):
        return info
    conditions_hint = hint.model_copy(update={"applies_when": conditions or None})
    return _merged(info, Field(json_schema_extra=conditions_hint.schema_extra()))


def _merged(info: FieldInfo, *changes: FieldInfo) -> FieldInfo:
    """``info`` with the attributes and constraints of ``changes`` merged in."""
    return FieldInfo.from_annotation(annotated(info.annotation, info, *changes))


def _type_of(default: Any) -> Any:
    """The type of an input that fills no setting, from its default."""
    for type_ in (bool, str, list):
        if isinstance(default, type_):
            return type_
    return float if isinstance(default, int | float) else JsonValue


def _validate_literal(
    value: Any, info: FieldInfo, path: Location, methods_by_ref: Mapping[str, Method[Any]]
) -> None:
    """Refuses a setting written as it is that its method wouldn't accept."""
    try:
        TypeAdapter[Any](annotated(info.annotation, *info.metadata)).validate_python(
            value, context={METHODS_CONTEXT_KEY: methods_by_ref}
        )
    except ValidationError as error:
        first_error = error.errors()[0]
        raise MethodFileFieldError(
            error_message(first_error), (*path, *first_error["loc"])
        ) from None


def input_field(declared_input: Input, annotation: Any, info: FieldInfo) -> tuple[Any, FieldInfo]:
    """An input's field: the setting it fills, with the input's own title, default and hints."""
    if declared_input.choices:
        annotation = literal(declared_input.choices)
    input_hints = {
        "advanced": declared_input.advanced,
        "labels": declared_input.labels,
        "applies_when": declared_input.applies_when,
    }
    hint = hint_of(info).model_copy(
        update={key: value for key, value in input_hints.items() if value is not None}
    )
    changes: dict[str, Any] = {
        "title": declared_input.title or info.title,
        "description": declared_input.description or info.description,
        "examples": declared_input.examples or info.examples,
        "json_schema_extra": hint.schema_extra() if hint != Hint() else None,
    }
    required = "default" not in declared_input.model_fields_set
    if required:
        changes["default_factory"] = None
    else:
        changes |= {
            "default": declared_input.default,
            "default_factory": None,
            "validate_default": True,
        }
    needs_item = get_origin(annotation) is list and declared_input.default != []
    field = _merged(info, *([Field(min_length=1)] if needs_item else []), Field(**changes))
    if required:  # merging keeps the setting's own default rather than an undefined one
        field.default = PydanticUndefined
    field.annotation = annotation
    return annotation, field


def settings_model(file: MethodFile, methods_by_ref: Mapping[str, Method[Any]]) -> type[Settings]:
    """The settings of the method ``file`` defines: one field for each of its inputs."""
    input_types = _InputTypes(file, methods_by_ref).inputs()
    fields = {
        key: input_field(declared_input, *input_types[key])
        for key, declared_input in file.inputs.items()
    }
    for key, declared_input in file.inputs.items():
        if "default" in declared_input.model_fields_set:
            _validate_literal(
                declared_input.default, fields[key][1], ("inputs", key, "default"), methods_by_ref
            )
    model: type[Settings] = create_model(file.label, __base__=Settings, **fields)  # type: ignore[call-overload]
    return model
