"""The templates of a method file, checked when it loads."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from ...common.paths import Location
from ..settings import Settings
from ..step import Step
from ..templates import (
    FEEDBACK_SCOPE,
    STEP_RESULT_SHAPE,
    ListOf,
    Shape,
    TemplateError,
    is_template,
    reference,
    template_errors,
    validate,
)
from .file import TEMPLATE_KEYS, MethodFile, MethodFileFieldError


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
            key: _template_shape(info.annotation) for key, info in annotation.model_fields.items()
        }
    return "any"


def validate_templates(file: MethodFile, settings: type[Settings]) -> None:
    declared_shapes = {
        key: _template_shape(info.annotation) for key, info in settings.model_fields.items()
    }
    templates: list[tuple[str, str | None, Mapping[str, Shape]]] = [
        ("description", file.description, {"settings": declared_shapes, "parts": ListOf("text")}),
        ("feedback", file.feedback, {"settings": declared_shapes, **FEEDBACK_SCOPE}),
    ]
    for name, source, scope in templates:
        if source is not None:
            try:
                validate(source, scope)
            except TemplateError as error:
                raise MethodFileFieldError(str(error), (name,)) from None
    if isinstance(file.steps, list):
        _validate_entries(file.steps, ("steps",), {"settings": declared_shapes}, {})


def _validate_entries(
    entries: list[Any],
    path: Location,
    scope: Mapping[str, Shape],
    earlier_results: Mapping[str, Shape],
) -> None:
    """The templates of a list of steps, each of which may read the results of those before."""
    earlier_results = dict(earlier_results)
    for index, entry in enumerate(entries):
        _validate_value(entry, (*path, index), scope, earlier_results)
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
            earlier_results[entry["id"]] = STEP_RESULT_SHAPE


def _validate_entry(
    entry: Mapping[str, Any],
    path: Location,
    scope: Mapping[str, Shape],
    earlier_results: Mapping[str, Shape],
) -> None:
    entry_scope = dict(scope)
    if "for_each" in entry:
        source = reference(entry["for_each"]) or ""
        declared_shapes = scope["settings"]
        declared_shape = (
            declared_shapes.get(source.removeprefix("settings."), "any")
            if isinstance(declared_shapes, Mapping)
            else "any"
        )
        entry_scope |= {
            "item": declared_shape.item if isinstance(declared_shape, ListOf) else "any",
            "index": "number",
        }
    for location, error in template_errors(entry, {**entry_scope, "steps": earlier_results}):
        raise MethodFileFieldError(str(error), (*path, *location))
    for key, value in entry.items():
        if key not in TEMPLATE_KEYS:
            _validate_value(value, (*path, key), entry_scope, earlier_results)


def _validate_value(
    value: Any, path: Location, scope: Mapping[str, Shape], earlier_results: Mapping[str, Shape]
) -> None:
    """The templates in ``value``: a step, or settings, which are filled in before running."""
    if isinstance(value, str) and is_template(value):
        try:
            validate(value, scope)
        except TemplateError as error:
            raise MethodFileFieldError(str(error), path) from None
    elif isinstance(value, Mapping) and "uses" in value:
        _validate_entry(value, path, scope, earlier_results)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _validate_value(item, (*path, key), scope, earlier_results)
    elif isinstance(value, list):
        _validate_entries(value, path, scope, earlier_results)
