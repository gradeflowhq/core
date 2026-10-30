"""The templates of a method file, checked when it loads."""

from collections.abc import Mapping
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from ...common.paths import Loc
from ..check import Check
from ..settings import Settings
from ..templates import (
    CHECK_RESULT,
    MARKED,
    ListOf,
    Shape,
    TemplateError,
    is_template,
    reference,
    template_errors,
    validate,
)
from .file import Fault, MethodFile


def _template_type(annotation: Any) -> Shape:
    """What a template sees of a value of ``annotation``, for checking names on load."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _template_type(args[0])
    if origin in (Union, UnionType):
        return _template_type(next(arg for arg in args if arg is not type(None)))
    if origin is list:
        return ListOf(_template_type(args[0]) if args else "any")
    if origin is Literal or annotation is str:
        return "text"
    if annotation is bool:
        return "bool"
    if annotation in (int, float):
        return "number"
    if isinstance(annotation, type) and issubclass(annotation, Check):
        return {"uses": "text", "name": "text", "question": "text", "weight": "number"}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {
            key: _template_type(info.annotation) for key, info in annotation.model_fields.items()
        }
    return "any"


def check_templates(file: MethodFile, settings: type[Settings]) -> None:
    inputs = {key: _template_type(info.annotation) for key, info in settings.model_fields.items()}
    templates: list[tuple[str, str | None, Mapping[str, Shape]]] = [
        ("description", file.description, {"inputs": inputs, "parts": ListOf("text")}),
        ("feedback", file.feedback, {"inputs": inputs, **MARKED}),
    ]
    for name, source, scope in templates:
        if source is not None:
            try:
                validate(source, scope)
            except TemplateError as error:
                raise Fault(str(error), (name,)) from None
    if isinstance(file.checks, list):
        _check_entries(file.checks, ("checks",), {"inputs": inputs}, {})


def _check_entries(
    entries: list[Any], path: Loc, scope: Mapping[str, Shape], earlier: Mapping[str, Shape]
) -> None:
    """The templates of a list of checks, each of which may read the results of those before."""
    earlier = dict(earlier)
    for index, entry in enumerate(entries):
        _check_value(entry, (*path, index), scope, earlier)
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
            earlier[entry["id"]] = CHECK_RESULT


def _check_entry(
    entry: Mapping[str, Any], path: Loc, scope: Mapping[str, Shape], earlier: Mapping[str, Shape]
) -> None:
    local = dict(scope)
    if "for_each" in entry:
        source = reference(entry["for_each"]) or ""
        inputs = scope["inputs"]
        kind = (
            inputs.get(source.removeprefix("inputs."), "any")
            if isinstance(inputs, Mapping)
            else "any"
        )
        local |= {"each": kind.item if isinstance(kind, ListOf) else "any", "index": "number"}
    for loc, error in template_errors(entry, {**local, "checks": earlier}):
        raise Fault(str(error), (*path, *loc))
    _check_value(entry.get("with"), (*path, "with"), local, earlier)


def _check_value(
    value: Any, path: Loc, scope: Mapping[str, Shape], earlier: Mapping[str, Shape]
) -> None:
    """The templates in ``value``: a check, or settings, which are filled in before marking."""
    if isinstance(value, str) and is_template(value):
        try:
            validate(value, scope)
        except TemplateError as error:
            raise Fault(str(error), path) from None
    elif isinstance(value, Mapping) and "uses" in value:
        _check_entry(value, path, scope, earlier)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _check_value(item, (*path, key), scope, earlier)
    elif isinstance(value, list):
        _check_entries(value, path, scope, earlier)
