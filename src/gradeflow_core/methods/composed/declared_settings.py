"""A composed method's declared settings, each typed by the setting it fills."""

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
from ..settings import Settings, annotated, item_field, literal
from ..step import METHODS_CONTEXT_KEY, STEP_OWN_FIELDS, step_cardinality
from ..templates import TemplateError, has_template, names_read_in, reference, validate
from .expand import EXPANDED_NAMES
from .file import TEMPLATE_KEYS, DeclaredSetting, MethodFile, MethodFileFieldError

if TYPE_CHECKING:
    from ..method import Method

_DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")
"""A reference by name, such as ``settings.points``."""


class _SettingTypes:
    """The type and field of each declared setting, from the setting it fills."""

    def __init__(self, file: MethodFile, methods_by_ref: Mapping[str, Method[Any]]) -> None:
        self.file = file
        self.methods_by_ref = methods_by_ref
        self.setting_types: dict[str, tuple[Any, FieldInfo]] = {}
        self.repeated_item_fields: list[dict[str, tuple[Any, FieldInfo]]] = []
        self.settings_in_templates = names_read_in([file.description, file.feedback], "settings")
        self.errors: list[MethodFileFieldError] = []
        """The problems found so far."""

    def declared_setting_types(self) -> dict[str, tuple[Any, FieldInfo]]:
        workflow = self.file.workflow()
        method = self.methods_by_ref.get(workflow["uses"])
        if method is None:
            raise MethodFileFieldError(f"There's no method {workflow['uses']}.", ("uses",))
        self._visit_settings(workflow["with"], method, ())
        for key, declared_setting in self.file.settings.items():
            with self._collect_errors():
                self._visit_declared_setting(key, declared_setting)
        if self.errors:
            raise MethodFileFieldError.combined(self.errors)
        return self.setting_types

    def _visit_declared_setting(self, key: str, declared_setting: DeclaredSetting) -> None:
        if key in STEP_OWN_FIELDS:
            raise MethodFileFieldError(
                f"Steps already have a field called {key}. Choose another name.", ("settings", key)
            )
        if (
            key in self.setting_types or self.errors
        ):  # a setting that failed may be the one that uses it
            return
        if key not in self.settings_in_templates:
            raise MethodFileFieldError(
                f"{declared_setting.title or f'Setting {key}'} isn't used.", ("settings", key)
            )
        self.setting_types[key] = (_type_of(declared_setting.default), FieldInfo())

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
            raise MethodFileFieldError("Use one {{ settings.name }} and nothing else.", path)
        root, *attribute_names = expression.split(".")
        if (
            root == "settings"
            and len(attribute_names) == 1
            and attribute_names[0] in self.file.settings
        ):
            existing = self.setting_types.get(attribute_names[0])
            if existing is not None and existing[0] != annotation:
                raise MethodFileFieldError(
                    f"Setting {attribute_names[0]} fills settings of different types.", path
                )
            self.setting_types.setdefault(attribute_names[0], (annotation, info))
        elif root == "item" and self.repeated_item_fields and len(attribute_names) <= 1:
            self.repeated_item_fields[-1].setdefault(
                attribute_names[0] if attribute_names else "", (annotation, info)
            )
        elif root != "index":
            raise MethodFileFieldError(f"There's no {expression}.", path)

    def _visit_step(self, node: Any, path: Location) -> None:
        if not isinstance(node, Mapping) or "uses" not in node:
            raise MethodFileFieldError("A step needs uses: and a method's name.", path)
        method = self.methods_by_ref.get(str(node["uses"]))
        if method is None:
            raise MethodFileFieldError(f"There's no method {node['uses']}.", (*path, "uses"))
        for key, value in sorted(node.items(), key=lambda item: item[0] != "with"):
            if key in TEMPLATE_KEYS:
                self.settings_in_templates |= names_read_in(value, "settings")
                continue
            if key in ("uses", "for_each"):
                continue
            if key == "with":
                self._visit_settings(value, method, (*path, "with"))
            elif key not in STEP_OWN_FIELDS:
                raise MethodFileFieldError(f"A step has no {key}:.", (*path, key))
            elif (expression := reference(value)) is not None:
                info = STEP_OWN_FIELDS[key]
                self._record(expression, info.annotation, info, (*path, key))

    def _visit_settings(self, values: Any, method: Method[Any], path: Location) -> None:
        if (expression := reference(values)) is not None:
            raise MethodFileFieldError(
                f"Fill in a method's settings one by one, not with {expression}.",
                path,
            )
        if not isinstance(values, Mapping | None):
            raise MethodFileFieldError("A step's settings aren't a mapping.", path)
        model = method.settings
        for key, value in (values or {}).items():
            with self._collect_errors():
                info = model.model_fields.get(key)
                if info is None:
                    raise MethodFileFieldError(
                        f"{method.label} has no setting {key}.", (*path, key)
                    )
                if (source := _source_of(value)) is not None:
                    info = _with_setting_conditions(info, source, values or {}, model, (*path, key))
                self._visit_setting(value, info, (*path, key))

    def _visit_setting(self, value: Any, info: FieldInfo, path: Location) -> None:
        if hint_of(info).kind == "output_step":  # read when the steps run
            return
        if (expression := reference(value)) is not None:
            self._record(expression, info.annotation, info, path)
            return
        match step_cardinality(info):
            case "one":
                self._visit_step(value, path)
            case "many" if (item_info := item_field(info)) is not None:
                for index, item in enumerate(value):
                    with self._collect_errors():
                        self._visit_item(item, item_info, (*path, index))
            case _ if has_template(value):
                self.settings_in_templates |= names_read_in(value, "settings")
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
            self._record(expression, info.rebuild_annotation(), info, path)
            return
        if isinstance(item, Mapping) and "for_each" in item:
            source = reference(item["for_each"])
            if source is None or not source.startswith("settings."):
                raise MethodFileFieldError(
                    "for_each needs one {{ settings.… }}.", (*path, "for_each")
                )
            self.repeated_item_fields.append({})
            self._visit_step(item, path)
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
                raise MethodFileFieldError("A repeated step uses {{ item }}.", path)
            self._record(
                source,
                list[annotation],  # type: ignore[valid-type]
                FieldInfo(title=field.title, json_schema_extra=field.json_schema_extra),
                (*path, "for_each"),
            )
            return
        self._visit_step(item, path)

    def _with_item_titles(
        self, item_fields: dict[str, tuple[Any, FieldInfo]], source: str, path: Location
    ) -> dict[str, tuple[Any, FieldInfo]]:
        """The fields of each item, with the titles the setting gives them."""
        declared_setting = self.file.settings.get(source.removeprefix("settings."))
        item_titles = (declared_setting.items if declared_setting else None) or {}
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
    """What fills a setting from elsewhere: ``settings``, or ``item`` of a repeated step."""
    expression = reference(value)
    if expression is None and isinstance(value, list):
        expression = next(
            (item_expression for item in value if (item_expression := reference(item))), None
        )
    return expression.split(".")[0] if expression else None


def _with_setting_conditions(
    info: FieldInfo,
    source: str,
    siblings: Mapping[str, Any],
    model: type[BaseModel],
    path: Location,
) -> FieldInfo:
    """``info`` filled from ``source``, its ``applies_when`` rewritten for the method file."""
    hint = hint_of(info)
    conditions: dict[str, list[JsonValue]] = {}
    for key, values in (hint.applies_when or {}).items():
        sibling = model.model_fields[key]
        value = siblings.get(key, sibling.get_default(call_default_factory=True))
        if (expression := reference(value)) is not None:
            scope, _, name = expression.partition(".")
            if scope == source and name:
                conditions[name] = values
        elif not has_template(value) and getattr(value, "value", value) not in values:
            labels = hint_of(sibling).labels or {}
            allowed_values = quoted_join(
                [labels.get(str(option), option) for option in values], conjunction="or"
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
    """The type of a declared setting that fills no other, from its default."""
    for type_ in (bool, str, list):
        if isinstance(default, type_):
            return type_
    return float if isinstance(default, int | float) else JsonValue


def _validate_literal(
    value: Any, info: FieldInfo, path: Location, methods_by_ref: Mapping[str, Method[Any]]
) -> None:
    """Refuses a setting written as it is that its method wouldn't accept."""
    try:
        TypeAdapter[Any](info.rebuild_annotation()).validate_python(
            value, context={METHODS_CONTEXT_KEY: methods_by_ref}
        )
    except ValidationError as error:
        first_error = error.errors()[0]
        raise MethodFileFieldError(
            error_message(first_error), (*path, *first_error["loc"])
        ) from None


def declared_setting_field(
    declared_setting: DeclaredSetting, annotation: Any, info: FieldInfo
) -> tuple[Any, FieldInfo]:
    """A declared setting's field: the setting it fills, with its own title, default and hints."""
    if declared_setting.choices:
        annotation = literal(declared_setting.choices)
    declared_hints = {
        "advanced": declared_setting.advanced,
        "labels": declared_setting.labels,
        "applies_when": declared_setting.applies_when,
    }
    hint = hint_of(info).model_copy(
        update={key: value for key, value in declared_hints.items() if value is not None}
    )
    changes: dict[str, Any] = {
        "title": declared_setting.title or info.title,
        "description": declared_setting.description or info.description,
        "examples": declared_setting.examples or info.examples,
        "json_schema_extra": hint.schema_extra() if hint != Hint() else None,
    }
    required = "default" not in declared_setting.model_fields_set
    if required:
        changes["default_factory"] = None
    else:
        changes |= {
            "default": declared_setting.default,
            "default_factory": None,
            "validate_default": True,
        }
    needs_item = get_origin(annotation) is list and declared_setting.default != []
    field = _merged(info, *([Field(min_length=1)] if needs_item else []), Field(**changes))
    if required:  # merging keeps the setting's own default rather than an undefined one
        field.default = PydanticUndefined
    field.annotation = annotation
    return annotation, field


def settings_model(file: MethodFile, methods_by_ref: Mapping[str, Method[Any]]) -> type[Settings]:
    """The settings model of the method ``file`` defines."""
    setting_types = _SettingTypes(file, methods_by_ref).declared_setting_types()
    fields = {
        key: declared_setting_field(declared_setting, *setting_types[key])
        for key, declared_setting in file.settings.items()
    }
    for key, declared_setting in file.settings.items():
        if "default" in declared_setting.model_fields_set:
            _validate_literal(
                declared_setting.default,
                fields[key][1],
                ("settings", key, "default"),
                methods_by_ref,
            )
    model: type[Settings] = create_model(file.label, __base__=Settings, **fields)  # type: ignore[call-overload]
    return model
