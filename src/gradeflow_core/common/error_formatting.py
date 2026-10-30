"""Plain-language messages for core errors."""

from typing import Any

from pydantic_core import ErrorDetails

from .text import as_text, sentence

_MAX_REASON_LENGTH = 500
_MESSAGES = {
    "missing": "Required.",
    "too_short": "Add at least {min_length}.",
    "too_long": "Keep at most {max_length}.",
    "int_parsing": "Must be a whole number.",
    "int_type": "Must be a whole number.",
    "int_from_float": "Must be a whole number.",
    "float_parsing": "Must be a number.",
    "float_type": "Must be a number.",
    "finite_number": "Must be a number.",
    "string_type": "Must be text.",
    "bool_parsing": "Must be true or false.",
    "bool_type": "Must be true or false.",
    "literal_error": "Choose {expected}.",
    "enum": "Choose {expected}.",
    "greater_than_equal": "Must be {ge} or more.",
    "greater_than": "Must be more than {gt}.",
    "less_than_equal": "Must be {le} or less.",
    "less_than": "Must be less than {lt}.",
    "list_type": "Must be a list.",
    "set_type": "Must be a list.",
    "dict_type": "Must be an object.",
    "model_type": "Must be an object.",
    "model_attributes_type": "Must be an object.",
    "union_tag_not_found": "Choose a type.",
    "extra_forbidden": "Not allowed here.",
    "string_pattern_mismatch": "Use letters, digits and underscores, not starting with a digit.",
}


def format_reason(reason: str) -> str:
    compact = " ".join(reason.split())
    if not compact:
        return "Unknown error."
    if len(compact) <= _MAX_REASON_LENGTH:
        return compact
    return f"{compact[: _MAX_REASON_LENGTH - 1]}..."


def error_message(error: ErrorDetails) -> str:
    """What is wrong with the value at the error's location, as a sentence that doesn't name it."""
    kind = error["type"]
    context: dict[str, Any] = error.get("ctx") or {}
    if kind == "union_tag_invalid":
        tag = context.get("tag")
        return "Choose a type." if tag in {None, "", "None"} else f"There's no type {tag}."
    if kind in {"value_error", "assertion_error"}:
        return sentence(format_reason(str(context.get("error") or error["msg"])))
    if kind == "too_short" and context.get("min_length") == 1:
        return "Add at least one."
    if kind == "string_too_short" and context.get("min_length") == 1:
        return _MESSAGES["missing"]
    if template := _MESSAGES.get(kind):
        return template.format(**{key: as_text(value) for key, value in context.items()})
    return sentence(format_reason(error["msg"]))
