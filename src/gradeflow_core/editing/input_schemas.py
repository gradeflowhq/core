"""The JSON schema of a composed method's input declaration, as the setting it fills allows it."""

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic.fields import FieldInfo

from ..common.hints import HINT_KEY, hint
from ..common.schemas import JsonDict, json_schema
from ..methods.composed.file import Input
from ..methods.composed.inputs import input_field

PLACEMENT_HINTS = ("advanced", "applies_when")
"""Hints of where a setting shows among its check's, which its input's own form doesn't follow."""
FROM_SETTING_KEYS = ("description", "examples")
"""What an input's default takes from the setting it fills, not from its own schema."""


def input_schema(default: JsonDict | None, setting_schema: JsonDict | None) -> JsonDict:
    """The JSON schema of an input's declaration whose default has the schema ``default``, filling
    the setting whose schema is ``setting_schema``, if any; each with the definitions it refers
    to."""
    schema = json_schema(Input)
    properties, definitions = schema["properties"], schema.setdefault("$defs", {})
    if default is not None:
        definitions.update(default.get("$defs", {}))
        default = {key: value for key, value in default.items() if key != "$defs"}
    setting = (
        _resolved(setting_schema, setting_schema.get("$defs", {}))
        if setting_schema
        else _resolved(default or {}, definitions)
    )
    labels = _labels(setting) | _labels(_resolved(default or {}, definitions))
    if default is not None:
        from_setting = {
            key: setting[key] for key in FROM_SETTING_KEYS if setting_schema and key in setting
        }
        properties["default"] = _default_property(
            default, definitions, setting, from_setting, labels
        )
    for key, property_schema in (
        ("choices", _choices(setting, labels)),
        ("examples", _examples(setting)),
    ):
        if property_schema is None:
            del properties[key]
        else:
            properties[key] = {"title": properties[key]["title"], **property_schema}
    properties["advanced"]["default"] = bool(_hint_values(setting).get("advanced"))
    return schema


def default_schema(declaration: Input, annotation: Any, field: FieldInfo) -> JsonDict:
    """The JSON schema of the default of the input ``declaration``, typed as the field it fills."""
    annotation, field = input_field(declaration, annotation, field)
    field.validate_default = None  # only for validating the default, not its schema
    return json_schema(Annotated[annotation, field])


def _resolved(schema: JsonDict, definitions: Mapping[str, JsonDict]) -> JsonDict:
    """``schema`` with its ``$ref`` to one of ``definitions`` filled in, without the null it may
    also be."""
    if "$ref" in schema:
        definition = definitions[schema["$ref"].rsplit("/", 1)[-1]]
        other_keys = {key: value for key, value in schema.items() if key != "$ref"}
        return _resolved({**definition, **other_keys}, definitions)
    unwrapped = _unwrapped(schema)
    return schema if unwrapped is schema else _resolved(unwrapped, definitions)


def _unwrapped(schema: JsonDict) -> JsonDict:
    """``schema`` without the null it may also be."""
    json_type = schema.get("type")
    if isinstance(json_type, list) and "null" in json_type:
        non_null_types = [each for each in json_type if each != "null"]
        return {
            **schema,
            "type": non_null_types[0] if len(non_null_types) == 1 else non_null_types,
        }
    options = schema.get("anyOf", [])
    non_null_options = [option for option in options if option.get("type") != "null"]
    if len(non_null_options) == 1 < len(options):
        return {
            **{key: value for key, value in schema.items() if key != "anyOf"},
            **non_null_options[0],
        }
    return schema


def _hint_values(schema: JsonDict) -> dict[str, Any]:
    hint_values: dict[str, Any] = schema.get(HINT_KEY, {})
    return hint_values


def _labels(schema: JsonDict) -> dict[str, str]:
    """The labels of a choice's options, or of a list of choices'."""
    labels: dict[str, str] = (
        _hint_values(schema).get("labels")
        or _hint_values(_unwrapped(schema.get("items", {}))).get("labels")
        or {}
    )
    return labels


def _default_property(
    default: JsonDict,
    definitions: JsonDict,
    setting: JsonDict,
    from_setting: JsonDict,
    labels: dict[str, str],
) -> JsonDict:
    """An input's default: one of the setting's options, if it has them, explained as it is."""
    if options := setting.get("enum"):
        default = {**_resolved(default, definitions), "enum": options}
    hints = {
        key: value for key, value in _hint_values(default).items() if key not in PLACEMENT_HINTS
    }
    return {
        **{key: value for key, value in default.items() if key not in FROM_SETTING_KEYS},
        **from_setting,
        "title": "Default",
        HINT_KEY: hints | ({"labels": labels} if labels else {}),
    }


def _choices(setting: JsonDict, labels: dict[str, str]) -> JsonDict | None:
    """An input's choices among a text setting's options; None for any other setting."""
    if not (options := setting.get("enum")) or setting.get("type") != "string":
        return None
    items = {"type": "string", "enum": options, **hint(labels=labels)}
    return {"type": "array", "uniqueItems": True, "items": items}


def _examples(setting: JsonDict) -> JsonDict | None:
    """An input's examples, typed like a text or number setting without options; None else."""
    if setting.get("enum") or (json_type := setting.get("type")) not in (
        "string",
        "number",
        "integer",
    ):
        return None
    return {"type": "array", "items": {"type": json_type}}
