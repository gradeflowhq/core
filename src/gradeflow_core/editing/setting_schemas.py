"""The JSON schema of a composed method's setting declaration, as the setting it fills allows it."""

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic.fields import FieldInfo

from ..common.hints import HINT_KEY, hint
from ..common.schemas import JsonDict, json_schema
from ..methods.composed.declared_settings import declared_setting_field
from ..methods.composed.file import DeclaredSetting

PLACEMENT_HINTS = ("advanced", "applies_when")
"""Hints that place a setting in its step's form, not in its own."""
INHERITED_KEYS = ("description", "examples")
"""What a declared setting's default takes from the setting it fills, not from its own schema."""


def declared_setting_schema(default: JsonDict | None, setting_schema: JsonDict | None) -> JsonDict:
    """The JSON schema of a declared setting, given the schemas of its default and of the
    setting it fills, each with its ``$defs``."""
    schema = json_schema(DeclaredSetting)
    properties, definitions = schema["properties"], schema.setdefault("$defs", {})
    if default is not None:
        definitions.update(default.get("$defs", {}))
        default = {key: value for key, value in default.items() if key != "$defs"}
    resolved_setting_schema = (
        _resolved(setting_schema, setting_schema.get("$defs", {}))
        if setting_schema
        else _resolved(default or {}, definitions)
    )
    labels = _labels(resolved_setting_schema) | _labels(_resolved(default or {}, definitions))
    if default is not None:
        inherited_entries = {
            key: resolved_setting_schema[key]
            for key in INHERITED_KEYS
            if setting_schema and key in resolved_setting_schema
        }
        properties["default"] = _default_property(
            default, definitions, resolved_setting_schema, inherited_entries, labels
        )
    for key, property_schema in (
        ("choices", _choices_schema(resolved_setting_schema, labels)),
        ("examples", _examples_schema(resolved_setting_schema)),
    ):
        if property_schema is None:
            del properties[key]
        else:
            properties[key] = {"title": properties[key]["title"], **property_schema}
    properties["advanced"]["default"] = bool(_hint_values(resolved_setting_schema).get("advanced"))
    return schema


def default_schema(declared_setting: DeclaredSetting, field: FieldInfo) -> JsonDict:
    """The JSON schema of the default of ``declared_setting``, typed as the field it fills."""
    annotation, field = declared_setting_field(declared_setting, field.annotation, field)
    field.validate_default = None  # only for validating the default, not its schema
    return json_schema(Annotated[annotation, field])


def _resolved(schema: JsonDict, definitions: Mapping[str, JsonDict]) -> JsonDict:
    """``schema`` with its ``$ref`` to one of ``definitions`` filled in, without the null it may
    also be."""
    if "$ref" in schema:
        definition = definitions[schema["$ref"].rsplit("/", 1)[-1]]
        schema_without_ref = {key: value for key, value in schema.items() if key != "$ref"}
        return _resolved({**definition, **schema_without_ref}, definitions)
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
    any_of_members = schema.get("anyOf", [])
    non_null_members = [member for member in any_of_members if member.get("type") != "null"]
    if len(non_null_members) == 1 < len(any_of_members):
        return {
            **{key: value for key, value in schema.items() if key != "anyOf"},
            **non_null_members[0],
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
    setting_schema: JsonDict,
    inherited_entries: JsonDict,
    labels: dict[str, str],
) -> JsonDict:
    """The schema of a declared setting's default, limited to the setting's options if any."""
    if options := setting_schema.get("enum"):
        default = {**_resolved(default, definitions), "enum": options}
    hints = {
        key: value for key, value in _hint_values(default).items() if key not in PLACEMENT_HINTS
    }
    return {
        **{key: value for key, value in default.items() if key not in INHERITED_KEYS},
        **inherited_entries,
        "title": "Default",
        HINT_KEY: hints | ({"labels": labels} if labels else {}),
    }


def _choices_schema(setting_schema: JsonDict, labels: dict[str, str]) -> JsonDict | None:
    """A declared setting's choices among a text setting's options; None for any other setting."""
    if not (options := setting_schema.get("enum")) or setting_schema.get("type") != "string":
        return None
    items = {"type": "string", "enum": options, **hint(labels=labels)}
    return {"type": "array", "uniqueItems": True, "items": items}


def _examples_schema(setting_schema: JsonDict) -> JsonDict | None:
    """Declared examples, typed like a text or number setting without options; None for any
    other setting."""
    if setting_schema.get("enum") or (json_type := setting_schema.get("type")) not in (
        "string",
        "number",
        "integer",
    ):
        return None
    return {"type": "array", "items": {"type": json_type}}
