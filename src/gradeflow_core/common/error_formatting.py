"""Plain-language messages for core errors."""

from typing import Any

from pydantic_core import ErrorDetails

from .text import as_sentence, as_text

_MAX_REASON_LENGTH = 500
IDENTIFIER_RULE = "Use letters, digits and underscores, not starting with a digit."
REQUIRED_MESSAGE = "Required."
UNKNOWN_MESSAGE = "Unknown setting."
_ERROR_MESSAGES = {
    "missing": REQUIRED_MESSAGE,
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
    "tuple_type": "Must be a list.",
    "dict_type": "Must be an object.",
    "model_type": "Must be an object.",
    "model_attributes_type": "Must be an object.",
    "union_tag_not_found": "Choose a type.",
    "extra_forbidden": UNKNOWN_MESSAGE,
    "string_pattern_mismatch": IDENTIFIER_RULE,
}


def shortened_reason(reason: str) -> str:
    compact_reason = " ".join(reason.split())
    if not compact_reason:
        return "Unknown error."
    if len(compact_reason) <= _MAX_REASON_LENGTH:
        return compact_reason
    return f"{compact_reason[: _MAX_REASON_LENGTH - 1]}..."


def error_message(error_details: ErrorDetails) -> str:
    """What is wrong with the value at the error's location, as a sentence that doesn't name it."""
    error_type = error_details["type"]
    context: dict[str, Any] = error_details.get("ctx") or {}
    if error_type == "union_tag_invalid":
        tag = context.get("tag")
        return "Choose a type." if tag in {None, "", "None"} else f"There's no type {tag}."
    if error_type in {"value_error", "assertion_error"}:
        return as_sentence(shortened_reason(str(context.get("error") or error_details["msg"])))
    if error_type == "too_short" and context.get("min_length") == 1:
        return "Add at least one."
    if error_type == "string_too_short" and context.get("min_length") == 1:
        return _ERROR_MESSAGES["missing"]
    if template := _ERROR_MESSAGES.get(error_type):
        return template.format(**{key: as_text(value) for key, value in context.items()})
    return as_sentence(shortened_reason(error_details["msg"]))
