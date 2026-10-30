"""The JSON schema of a composed method's input declaration, as the setting it fills allows it."""

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic.fields import FieldInfo

from ..common.hints import HINT_KEY, hint
from ..common.schemas import JsonDict, json_schema
from ..methods.composed.file import Input
from ..methods.composed.inputs import input_field

PLACEMENT = ("advanced", "applies_when")
"""Hints of where a setting shows among its check's, which its input's own form doesn't follow."""
EXPLAINED = ("description", "examples")
"""What an input's default takes from the setting it fills, not from its own schema."""


def input_schema(default: JsonDict | None, filled: JsonDict | None) -> JsonDict:
    """The JSON schema of an input's declaration whose default has the schema ``default``, filling
    the setting whose schema is ``filled``, if any; each with the definitions it refers to."""
    schema = json_schema(Input)
    properties, defs = schema["properties"], schema.setdefault("$defs", {})
    if default is not None:
        defs.update(default.get("$defs", {}))
        default = {key: value for key, value in default.items() if key != "$defs"}
    setting = (
        _resolved(filled, filled.get("$defs", {})) if filled else _resolved(default or {}, defs)
    )
    labels = _labels(setting) | _labels(_resolved(default or {}, defs))
    if default is not None:
        explained = {key: setting[key] for key in EXPLAINED if filled and key in setting}
        properties["default"] = _default_property(default, defs, setting, explained, labels)
    for key, made in (("choices", _choices(setting, labels)), ("examples", _examples(setting))):
        if made is None:
            del properties[key]
        else:
            properties[key] = {"title": properties[key]["title"], **made}
    properties["advanced"]["default"] = bool(_hint(setting).get("advanced"))
    return schema


def default_schema(spec: Input, annotation: Any, info: FieldInfo) -> JsonDict:
    """The JSON schema of the default of the input ``spec``, typed as the field it fills."""
    annotation, info = input_field(spec, annotation, info)
    info.validate_default = None  # only for validating the default, not its schema
    return json_schema(Annotated[annotation, info])


def _resolved(schema: JsonDict, defs: Mapping[str, JsonDict]) -> JsonDict:
    """``schema`` with its ``$ref`` to one of ``defs`` filled in, without the null it may also
    be."""
    if "$ref" in schema:
        found = defs[schema["$ref"].rsplit("/", 1)[-1]]
        rest = {key: value for key, value in schema.items() if key != "$ref"}
        return _resolved({**found, **rest}, defs)
    unwrapped = _unwrapped(schema)
    return schema if unwrapped is schema else _resolved(unwrapped, defs)


def _unwrapped(schema: JsonDict) -> JsonDict:
    """``schema`` without the null it may also be."""
    kind = schema.get("type")
    if isinstance(kind, list) and "null" in kind:
        rest = [each for each in kind if each != "null"]
        return {**schema, "type": rest[0] if len(rest) == 1 else rest}
    options = schema.get("anyOf", [])
    others = [option for option in options if option.get("type") != "null"]
    if len(others) == 1 < len(options):
        return {**{key: value for key, value in schema.items() if key != "anyOf"}, **others[0]}
    return schema


def _hint(schema: JsonDict) -> dict[str, Any]:
    found: dict[str, Any] = schema.get(HINT_KEY, {})
    return found


def _labels(schema: JsonDict) -> dict[str, str]:
    """The labels of a choice's options, or of a list of choices'."""
    labels: dict[str, str] = (
        _hint(schema).get("labels")
        or _hint(_unwrapped(schema.get("items", {}))).get("labels")
        or {}
    )
    return labels


def _default_property(
    default: JsonDict,
    defs: JsonDict,
    setting: JsonDict,
    explained: JsonDict,
    labels: dict[str, str],
) -> JsonDict:
    """An input's default: one of the setting's options, if it has them, explained as it is."""
    if options := setting.get("enum"):
        default = {**_resolved(default, defs), "enum": options}
    hints = {key: value for key, value in _hint(default).items() if key not in PLACEMENT}
    return {
        **{key: value for key, value in default.items() if key not in EXPLAINED},
        **explained,
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
    if setting.get("enum") or (kind := setting.get("type")) not in ("string", "number", "integer"):
        return None
    return {"type": "array", "items": {"type": kind}}
