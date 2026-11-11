"""The templates of a method file, checked when it loads."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from ...common.paths import Location
from ...steps.settings import Settings
from ...steps.step import Step
from ...steps.templates import (
    FEEDBACK_SCOPE,
    STEP_RESULT_SHAPE,
    ListOf,
    Shape,
    TemplateError,
    is_template,
    reference_expression,
    template_errors,
    validate,
)
from .file import TEMPLATE_KEYS, MethodFile, method_file_error


def _template_shape(annotation: Any) -> Shape:
    """What a template sees of a value of ``annotation``, for checking names on load."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _template_shape(args[0])
    if origin is Union:
        return _template_shape(next(arg for arg in args if arg is not type(None)))
    if origin is list:
        return ListOf(_template_shape(args[0]) if args else "any")
    if origin is Literal or annotation is str:
        return "text"
    if annotation is bool:
        return "bool"
    if annotation in (int, float):
        return "number"
    if isinstance(annotation, type) and issubclass(annotation, Step):
        return {"uses": "text", "name": "text", "question": "text", "weight": "number"}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {
            field_name: _template_shape(field_info.annotation)
            for field_name, field_info in annotation.model_fields.items()
        }
    return "any"


def validate_templates(file: MethodFile, settings: type[Settings]) -> None:
    declared_shapes = {
        setting_name: _template_shape(field_info.annotation)
        for setting_name, field_info in settings.model_fields.items()
    }
    templates: list[tuple[str, str | None, Mapping[str, Shape]]] = [
        ("description", file.description, {"settings": declared_shapes, "parts": ListOf("text")}),
        ("feedback", file.feedback, {"settings": declared_shapes, **FEEDBACK_SCOPE}),
    ]
    for name, template, scope in templates:
        if template is not None:
            try:
                validate(template, scope)
            except TemplateError as error:
                raise method_file_error(str(error), (name,)) from None
    if isinstance(file.steps, list):
        _validate_entries(file.steps, ("steps",), {"settings": declared_shapes}, {})


def _validate_entries(
    entries: list[Any],
    path: Location,
    scope: Mapping[str, Shape],
    earlier_step_shapes: Mapping[str, Shape],
) -> None:
    """The templates of a list of steps, each of which may read the results of those before."""
    earlier_step_shapes = dict(earlier_step_shapes)
    for index, entry in enumerate(entries):
        _validate_value(entry, (*path, index), scope, earlier_step_shapes)
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
            earlier_step_shapes[entry["id"]] = STEP_RESULT_SHAPE


def _validate_step_data(
    step_data: Mapping[str, Any],
    path: Location,
    scope: Mapping[str, Shape],
    earlier_step_shapes: Mapping[str, Shape],
) -> None:
    step_scope = dict(scope)
    if "for_each" in step_data:
        list_reference = reference_expression(step_data["for_each"]) or ""
        declared_shapes = scope["settings"]
        declared_shape = (
            declared_shapes.get(list_reference.removeprefix("settings."), "any")
            if isinstance(declared_shapes, Mapping)
            else "any"
        )
        step_scope |= {
            "item": declared_shape.item if isinstance(declared_shape, ListOf) else "any",
            "index": "number",
        }
    for location, error in template_errors(step_data, {**step_scope, "steps": earlier_step_shapes}):
        raise method_file_error(str(error), (*path, *location))
    for step_key, value in step_data.items():
        if step_key == "with" and isinstance(value, Mapping) and "output_step" in value:
            value = dict(value)
            steps = value.get("steps")
            nested_step_shapes = {
                nested_step_data["id"]: STEP_RESULT_SHAPE
                for nested_step_data in (steps if isinstance(steps, list) else ())
                if isinstance(nested_step_data, Mapping)
                and isinstance(nested_step_data.get("id"), str)
            }
            _validate_value(
                value.pop("output_step"),
                (*path, step_key, "output_step"),
                {**step_scope, "steps": nested_step_shapes},
                earlier_step_shapes,
            )
        if step_key not in TEMPLATE_KEYS:
            _validate_value(value, (*path, step_key), step_scope, earlier_step_shapes)


def _validate_value(
    value: Any, path: Location, scope: Mapping[str, Shape], earlier_step_shapes: Mapping[str, Shape]
) -> None:
    """The templates in ``value``: a step, or settings, which are filled in before running."""
    if isinstance(value, str) and is_template(value):
        try:
            validate(value, scope)
        except TemplateError as error:
            raise method_file_error(str(error), path) from None
    elif isinstance(value, Mapping) and "uses" in value:
        _validate_step_data(value, path, scope, earlier_step_shapes)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _validate_value(item, (*path, key), scope, earlier_step_shapes)
    elif isinstance(value, list):
        _validate_entries(value, path, scope, earlier_step_shapes)
