"""A composed method's declared settings, each typed by the setting it fills."""

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, get_args, get_origin

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ...common.hints import Hint, hint_of
from ...common.paths import pointer
from ...common.problems import InvalidError, Problem, validation_problem
from ...common.text import quoted_join
from ...steps.method import Method
from ...steps.settings import annotated, item_field, literal_type, step_cardinality
from ...steps.step import METHODS_CONTEXT_KEY, NOT_A_STEP_MESSAGE, STEP_OWN_PROPERTIES
from ...steps.templates import (
    TemplateError,
    has_template,
    names_read_in,
    reference_expression,
    validate,
)
from .expand import EXPANDED_NAMES, ComposedSettings
from .file import TEMPLATE_KEYS, DeclaredSetting, MethodFile, method_file_error

_DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")
"""A reference by name, such as ``settings.points``."""


class _SettingTypes:
    """The type and field of each declared setting, from the setting it fills."""

    def __init__(self, file: MethodFile, methods_by_ref: Mapping[str, Method]) -> None:
        self.file = file
        self.methods_by_ref = methods_by_ref
        self.setting_types: dict[str, tuple[Any, FieldInfo]] = {}
        self.repeated_item_fields: list[dict[str, tuple[Any, FieldInfo]]] = []
        self.settings_in_templates = names_read_in(
            [file.description, file.feedback, file.steps, file.marks_by], "settings"
        )
        self.problems: list[Problem] = []
        """The problems found so far."""

    def declared_setting_types(self) -> dict[str, tuple[Any, FieldInfo]]:
        workflow_step_data = self.file.workflow_step_data()
        method = self.methods_by_ref.get(workflow_step_data["uses"])
        if method is None:
            raise method_file_error(f"There's no method {workflow_step_data['uses']}.", "/uses")
        self._visit_settings(workflow_step_data["with"], method, "")
        for setting_name, declared_setting in self.file.settings.items():
            with self._collect_errors():
                self._visit_declared_setting(setting_name, declared_setting)
        if self.problems:
            raise InvalidError(self.problems)
        return self.setting_types

    def _visit_declared_setting(self, setting_name: str, declared_setting: DeclaredSetting) -> None:
        if message := setting_name_problem(setting_name):
            raise method_file_error(message, pointer("settings", setting_name))
        # A setting that failed may be the one that uses it.
        if setting_name in self.setting_types or self.problems:
            return
        if setting_name not in self.settings_in_templates:
            raise method_file_error(
                f"{declared_setting.title or f'Setting {setting_name}'} isn't used.",
                pointer("settings", setting_name),
            )
        self.setting_types[setting_name] = (_type_of(declared_setting.default), FieldInfo())

    @contextmanager
    def _collect_errors(self) -> Iterator[None]:
        """Records a problem raised inside, and goes on."""
        try:
            yield
        except InvalidError as error:
            self.problems += error.problems

    def _record_setting_type(
        self, expression: str, annotation: Any, field_info: FieldInfo, path: str
    ) -> None:
        if not _DOTTED_NAME.fullmatch(expression):
            # A template's own refusal says best what's wrong, such as arithmetic.
            try:
                validate(
                    expression, dict.fromkeys(EXPANDED_NAMES, "any"), expression_kind="reference"
                )
            except TemplateError as error:
                raise method_file_error(str(error), path) from None
            raise method_file_error("Use one {{ settings.name }} and nothing else.", path)
        root, *attribute_names = expression.split(".")
        if (
            root == "settings"
            and len(attribute_names) == 1
            and attribute_names[0] in self.file.settings
        ):
            recorded_type = self.setting_types.get(attribute_names[0])
            if recorded_type is not None and recorded_type[0] != annotation:
                raise method_file_error(
                    f"Setting {attribute_names[0]} fills settings of different types.", path
                )
            self.setting_types.setdefault(attribute_names[0], (annotation, field_info))
        elif root == "item" and self.repeated_item_fields and len(attribute_names) <= 1:
            self.repeated_item_fields[-1].setdefault(
                attribute_names[0] if attribute_names else "", (annotation, field_info)
            )
        elif root != "index":
            raise method_file_error(f"There's no {expression}.", path)

    def _visit_step(self, step_data: Any, path: str) -> None:
        if not isinstance(step_data, Mapping) or "uses" not in step_data:
            raise method_file_error(NOT_A_STEP_MESSAGE, path)
        method = self.methods_by_ref.get(str(step_data["uses"]))
        if method is None:
            raise method_file_error(f"There's no method {step_data['uses']}.", path + "/uses")
        for step_key, value in sorted(step_data.items(), key=lambda item: item[0] != "with"):
            if step_key in TEMPLATE_KEYS or step_key in ("uses", "for_each"):
                continue
            if step_key == "with":
                self._visit_settings(value, method, path + "/with")
            elif step_key not in STEP_OWN_PROPERTIES:
                raise method_file_error(f"A step has no {step_key}:.", path + pointer(step_key))
            elif (expression := reference_expression(value)) is not None:
                field_info = STEP_OWN_PROPERTIES[step_key]
                self._record_setting_type(
                    expression, field_info.annotation, field_info, path + pointer(step_key)
                )

    def _visit_settings(self, settings: Any, method: Method, path: str) -> None:
        if (expression := reference_expression(settings)) is not None:
            raise method_file_error(
                f"Fill in a method's settings one by one, not with {expression}.",
                path,
            )
        if not isinstance(settings, Mapping | None):
            raise method_file_error("A step's settings aren't a mapping.", path)
        model = method.settings
        for setting_name, value in (settings or {}).items():
            with self._collect_errors():
                field_info = model.model_fields.get(setting_name)
                if field_info is None:
                    raise method_file_error(
                        f"{method.label} has no setting {setting_name}.",
                        path + pointer(setting_name),
                    )
                if (root := _root_of(value)) is not None:
                    field_info = _with_setting_conditions(
                        field_info, root, settings or {}, model, path + pointer(setting_name)
                    )
                self._visit_setting(value, field_info, path + pointer(setting_name))

    def _visit_setting(self, value: Any, field_info: FieldInfo, path: str) -> None:
        if hint_of(field_info).kind == "output_step":  # read when the steps run
            return
        if (expression := reference_expression(value)) is not None:
            self._record_setting_type(expression, field_info.annotation, field_info, path)
            return
        match step_cardinality(field_info):
            case "one":
                self._visit_step(value, path)
            case "many" if (
                isinstance(value, list) and (item_field_info := item_field(field_info)) is not None
            ):
                for index, item in enumerate(value):
                    with self._collect_errors():
                        self._visit_item(item, item_field_info, path + pointer(index))
            case _ if has_template(value):
                item_annotation = (
                    get_args(field_info.annotation)[0] if get_args(field_info.annotation) else Any
                )
                for index, item in enumerate(value if isinstance(value, list) else ()):
                    if (expression := reference_expression(item)) is not None:
                        self._record_setting_type(
                            expression, item_annotation, field_info, path + pointer(index)
                        )
                    elif not has_template(item) and (item_field_info := item_field(field_info)):
                        _validate_written_value(
                            item, item_field_info, path + pointer(index), self.methods_by_ref
                        )
            case _:
                _validate_written_value(value, field_info, path, self.methods_by_ref)

    def _visit_item(self, item: Any, field_info: FieldInfo, path: str) -> None:
        if (expression := reference_expression(item)) is not None:
            self._record_setting_type(expression, field_info.rebuild_annotation(), field_info, path)
            return
        if isinstance(item, Mapping) and "for_each" in item:
            list_reference = reference_expression(item["for_each"])
            if list_reference is None or not list_reference.startswith("settings."):
                raise method_file_error("for_each needs one {{ settings.… }}.", path + "/for_each")
            self.repeated_item_fields.append({})
            self._visit_step(item, path)
            item_fields = self._with_item_titles(
                self.repeated_item_fields.pop(), list_reference, path
            )
            if "" in item_fields:
                annotation, item_field_info = item_fields[""]
                annotation = annotated(annotation, *item_field_info.metadata)
            elif item_fields:
                annotation = create_model(
                    "Item",
                    **{
                        item_field_name: (annotation, item_field_info)
                        for item_field_name, (annotation, item_field_info) in item_fields.items()
                    },
                )  # type: ignore[call-overload]
                item_field_info = FieldInfo()
            else:
                raise method_file_error("A repeated step uses {{ item }}.", path)
            self._record_setting_type(
                list_reference,
                list[annotation],  # type: ignore[valid-type]
                FieldInfo(
                    title=item_field_info.title, json_schema_extra=item_field_info.json_schema_extra
                ),
                path + "/for_each",
            )
            return
        self._visit_step(item, path)

    def _with_item_titles(
        self, item_fields: dict[str, tuple[Any, FieldInfo]], list_reference: str, path: str
    ) -> dict[str, tuple[Any, FieldInfo]]:
        """The fields of each item, with the titles the setting gives them."""
        declared_setting = self.file.settings.get(list_reference.removeprefix("settings."))
        item_titles = (declared_setting.items if declared_setting else None) or {}
        if unused_item_fields := item_titles.keys() - item_fields.keys():
            raise method_file_error(
                f"Each item has no {', '.join(sorted(unused_item_fields))}.", path + "/for_each"
            )
        return {
            item_field_name: (
                annotation,
                _merged(
                    item_field_info,
                    Field(**item_titles[item_field_name].model_dump(exclude_none=True)),
                ),
            )
            if item_field_name in item_titles
            else (annotation, item_field_info)
            for item_field_name, (annotation, item_field_info) in item_fields.items()
        }


def setting_name_problem(name: str) -> str | None:
    """Why a declared setting can't be called ``name``, if it can't: a step has a field of that
    name, or the settings model has it already, as it has ``expansion``."""
    if name in STEP_OWN_PROPERTIES:
        return f"Steps already have a field called {name}. Choose another name."
    # Pydantic keeps names with a leading underscore for private attributes.
    if name.startswith("_") or hasattr(ComposedSettings, name):
        return f"A setting can't be called {name}. Choose another name."
    return None


def _root_of(value: Any) -> str | None:
    """What fills a setting from elsewhere: ``settings``, or ``item`` of a repeated step."""
    expression = reference_expression(value)
    if expression is None and isinstance(value, list):
        expression = next(
            (item_expression for item in value if (item_expression := reference_expression(item))),
            None,
        )
    return expression.split(".")[0] if expression else None


def _with_setting_conditions(
    field_info: FieldInfo,
    root: str,
    sibling_settings: Mapping[str, Any],
    model: type[BaseModel],
    path: str,
) -> FieldInfo:
    """``field_info`` filled from ``root``, its ``applies_when`` rewritten for the method file."""
    hint = hint_of(field_info)
    conditions: dict[str, list[JsonValue]] = {}
    for sibling_name, applicable_values in (hint.applies_when or {}).items():
        sibling_field_info = model.model_fields[sibling_name]
        value = sibling_settings.get(
            sibling_name, sibling_field_info.get_default(call_default_factory=True)
        )
        if (expression := reference_expression(value)) is not None:
            scope, _, name = expression.partition(".")
            if scope == root and name:
                conditions[name] = applicable_values
        elif not has_template(value) and getattr(value, "value", value) not in applicable_values:
            labels = hint_of(sibling_field_info).labels or {}
            allowed_values = quoted_join(
                [labels.get(str(option), option) for option in applicable_values], conjunction="or"
            )
            raise method_file_error(
                f"{field_info.title} applies when {sibling_field_info.title} is {allowed_values}.",
                path,
            )
    if conditions == (hint.applies_when or {}):
        return field_info
    conditions_hint = hint.model_copy(update={"applies_when": conditions or None})
    return _merged(field_info, Field(json_schema_extra=conditions_hint.schema_extra()))


def _merged(field_info: FieldInfo, *changes: FieldInfo) -> FieldInfo:
    """``field_info`` with the attributes and constraints of ``changes`` merged in."""
    return FieldInfo.from_annotation(annotated(field_info.annotation, field_info, *changes))


def _type_of(default: Any) -> Any:
    """The type of a declared setting that fills no other, from its default."""
    for type_ in (bool, str, list):
        if isinstance(default, type_):
            return type_
    return float if isinstance(default, int | float) else JsonValue


def _validate_written_value(
    value: Any,
    field_info: FieldInfo,
    path: str,
    methods_by_ref: Mapping[str, Method],
) -> None:
    """Refuses a setting written as it is that its method wouldn't accept."""
    try:
        TypeAdapter[Any](field_info.rebuild_annotation()).validate_python(
            value, context={METHODS_CONTEXT_KEY: methods_by_ref}
        )
    except ValidationError as error:
        problem = validation_problem(error.errors()[0], value)
        raise method_file_error(problem.message, path + problem.field) from None


def declared_setting_field(
    declared_setting: DeclaredSetting, annotation: Any, field_info: FieldInfo
) -> tuple[Any, FieldInfo]:
    """A declared setting's field: the setting it fills, with its own title, default and hints."""
    if declared_setting.choices:
        annotation = literal_type(declared_setting.choices)
    declared_hints = {
        "advanced": declared_setting.advanced,
        "labels": declared_setting.labels,
        "applies_when": declared_setting.applies_when,
    }
    hint = hint_of(field_info).model_copy(
        update={key: value for key, value in declared_hints.items() if value is not None}
    )
    changes: dict[str, Any] = {
        "title": declared_setting.title or field_info.title,
        "description": declared_setting.description or field_info.description,
        "examples": declared_setting.examples or field_info.examples,
        "json_schema_extra": hint.schema_extra() if hint != Hint() else None,
        "default_factory": None,
    }
    is_required = "default" not in declared_setting.model_fields_set
    if not is_required:
        changes |= {"default": declared_setting.default, "validate_default": True}
    needs_item = get_origin(annotation) is list and declared_setting.default != []
    declared_field_info = _merged(
        field_info, *([Field(min_length=1)] if needs_item else []), Field(**changes)
    )
    if is_required:  # merging keeps the setting's own default rather than an undefined one
        declared_field_info.default = PydanticUndefined
    declared_field_info.annotation = annotation
    return annotation, declared_field_info


def settings_model(
    file: MethodFile, methods_by_ref: Mapping[str, Method]
) -> type[ComposedSettings]:
    """The settings model of the method ``file`` defines."""
    setting_types = _SettingTypes(file, methods_by_ref).declared_setting_types()
    fields = {
        setting_name: declared_setting_field(declared_setting, *setting_types[setting_name])
        for setting_name, declared_setting in file.settings.items()
    }
    for setting_name, declared_setting in file.settings.items():
        if "default" in declared_setting.model_fields_set:
            _validate_written_value(
                declared_setting.default,
                fields[setting_name][1],
                pointer("settings", setting_name, "default"),
                methods_by_ref,
            )
    model: type[ComposedSettings] = create_model(  # type: ignore[call-overload]
        file.label,
        __base__=ComposedSettings,
        __cls_kwargs__={"workflow": file.workflow_step_data()},
        **fields,
    )
    return model
