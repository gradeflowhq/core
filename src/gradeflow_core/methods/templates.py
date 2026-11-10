"""Sandboxed Jinja for references, conditions and templates, with an allowlist checked on load."""

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Literal, get_type_hints

from jinja2 import StrictUndefined, Template, TemplateSyntaxError, meta, nodes
from jinja2.environment import TemplateExpression
from jinja2.sandbox import SandboxedEnvironment

from ..common.exceptions import GradeFlowError
from ..common.text import (
    code_span,
    escaped_markdown,
    format_number,
    markdown_list,
    plain_join,
    plain_markdown,
    quoted,
    reads_better_listed,
)
from .outcome import CheckOutput, PartResult

REFERENCE = re.compile(r"^\{\{((?:(?!\{\{|\}\}).)+)\}\}$", re.DOTALL)
"""A value that is exactly one reference, which keeps the type of what it refers to."""


def _per_item(write: Callable[[Any], str]) -> Callable[[Any], Any]:
    """``write`` as a filter of a value or of each item of a list."""
    return lambda value: (
        [write(item) for item in value] if isinstance(value, list | tuple) else write(value)
    )


ENVIRONMENT = SandboxedEnvironment(
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    finalize=lambda value: format_number(value) if isinstance(value, float) else value,
)
_GRADEFLOW_FILTERS: dict[str, Callable[..., Any]] = {
    "and_list": partial(plain_join, conjunction="and"),
    "or_list": partial(plain_join, conjunction="or"),
    "quoted": _per_item(quoted),
    "code": _per_item(code_span),
    "escaped": _per_item(escaped_markdown),
    "needs_list": reads_better_listed,
    "percent": lambda value: f"{value:.0%}",
    "description_list": lambda checks: markdown_list(
        check.described_in_list(()) for check in checks
    ),
    "description": lambda check: plain_markdown(check.describe()),
}
ENVIRONMENT.filters.update(_GRADEFLOW_FILTERS)
ALLOWED_FILTERS = frozenset(
    {
        *_GRADEFLOW_FILTERS,
        "selectattr",
        "rejectattr",
        "map",
        "list",
        "length",
        "first",
        "last",
        "join",
        "lower",
        "upper",
        "capitalize",
        "round",
    }
)
_ALLOWED_NODES = (
    nodes.Template,
    nodes.Output,
    nodes.TemplateData,
    nodes.If,
    nodes.For,
    nodes.Name,
    nodes.Getattr,
    nodes.Getitem,
    nodes.Const,
    nodes.List,
    nodes.Compare,
    nodes.Operand,
    nodes.And,
    nodes.Or,
    nodes.Not,
    nodes.Neg,
    nodes.Filter,
    nodes.Keyword,
    nodes.CondExpr,
    nodes.Concat,
)

_STATEMENT_KEYWORDS = {
    "assign": "set",
    "assignblock": "set",
    "callblock": "call",
    "filterblock": "filter",
    "fromimport": "import",
    "scope": "with",
    "scopedevalcontextmodifier": "autoescape",
}
"""The statements refused nodes are written as, where their names differ."""

type Scalar = Literal["text", "number", "bool", "any"]
type Shape = Scalar | ListOf | Mapping[str, Shape]
"""What a template sees of a value, to check the names it reads: an object's fields by name."""


@dataclass(frozen=True)
class ListOf:
    item: Shape


_SCALARS: Mapping[type, Scalar] = {bool: "bool", float: "number", str: "text"}


def _fields_shape(typed_dict: type) -> Mapping[str, Shape]:
    return {key: _SCALARS[annotation] for key, annotation in get_type_hints(typed_dict).items()}


CHECK_OUTPUT_SHAPE = _fields_shape(CheckOutput)
CHECK_RESULT_SHAPE: Mapping[str, Shape] = {"output": CHECK_OUTPUT_SHAPE, "skipped": "bool"}
"""An earlier check, as ``steps.<id>``."""
STEP_RESULT_SHAPE: Mapping[str, Shape] = {"output": "any", "skipped": "bool"}
"""An earlier step whose kind isn't known, or a transform, as ``steps.<id>``."""
PART_RESULT_SHAPE = _fields_shape(PartResult)
"""A nested check's result, as an item of ``parts``."""
LOOP_VARIABLES: Mapping[str, Shape] = {
    "index": "number",
    "first": "bool",
    "last": "bool",
    "length": "number",
}
FEEDBACK_SCOPE: Mapping[str, Shape] = {
    "answer": "text",
    "correct": "bool",
    "proportion": "number",
    "feedback": "text",
    "parts": ListOf(PART_RESULT_SHAPE),
}
"""What a check's feedback template sees of how it marked: `feedback` is its method's own."""


class TemplateError(GradeFlowError, ValueError):
    """A template, condition or reference that isn't allowed, or names something not in scope."""


def reference_expression(value: Any) -> str | None:
    """The expression of a value that is exactly one ``{{ … }}``."""
    if isinstance(value, str) and (match := REFERENCE.fullmatch(value.strip())):
        return match[1].strip()
    return None


@lru_cache(maxsize=4096)
def expression_node(text: str) -> nodes.Expr | None:
    """The expression of ``text`` when it is exactly one ``{{ … }}``, parsed."""
    try:
        body = ENVIRONMENT.parse(text.strip()).body
    except TemplateSyntaxError:
        return None
    match body:
        case [nodes.Output(nodes=[expression])] if not isinstance(expression, nodes.TemplateData):
            return expression
    return None


def condition_expression(text: str) -> str:
    """The expression of an ``if``, which must be exactly one ``{{ … }}``."""
    if (expression := reference_expression(text)) is None:
        raise TemplateError(
            "Write a condition as one {{ … }}, such as {{ steps.test.output.correct }}."
        )
    return expression


def top_level_names(expression: str) -> frozenset[str]:
    """The names ``expression`` reads, such as ``settings`` and ``steps``."""
    return frozenset(meta.find_undeclared_variables(ENVIRONMENT.parse(f"{{{{ {expression} }}}}")))


def is_template(text: str) -> bool:
    """Whether ``text`` has a reference or a statement in it, so it is filled in when it marks."""
    return "{{" in text or "{%" in text


def has_template(value: Any) -> bool:
    """Whether ``value`` has a reference or template in it, so it is only known when it marks."""
    return any(map(is_template, _strings(value)))


def _strings(value: Any) -> Iterator[str]:
    """The strings anywhere in ``value``, as YAML holds it."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def validate(
    source: str,
    scope: Mapping[str, Shape],
    *,
    expression_kind: Literal["reference", "condition"] | None = None,
) -> None:
    """Refuses a template, or with ``expression_kind`` a bare expression, that uses anything not
    allowed or not in ``scope``, the types of what it may name."""
    try:
        tree = ENVIRONMENT.parse(f"{{{{ {source} }}}}" if expression_kind else source)
    except TemplateSyntaxError as error:
        if expression_kind:
            raise TemplateError(f"Can't read this {expression_kind}.") from None
        raise TemplateError(f"{error.message} (line {error.lineno})") from None
    _validate_node(tree, scope)


def _validate_node(node: nodes.Node, scope: Mapping[str, Shape]) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        raise _refusal_error(node)
    if isinstance(node, nodes.Getattr) and node.attr.startswith("_"):
        raise TemplateError(f"There's no {node.attr}.")
    if isinstance(node, nodes.For):
        _validate_loop(node, scope)
        return
    if isinstance(node, nodes.Filter) and node.name not in ALLOWED_FILTERS:
        raise TemplateError(f"There's no filter {node.name}.")
    if isinstance(node, nodes.Name | nodes.Getattr | nodes.Getitem):
        _shape_of(node, scope)
    for child in node.iter_child_nodes():
        _validate_node(child, scope)


def _refusal_error(node: nodes.Node) -> TemplateError:
    name = type(node).__name__.lower()
    if isinstance(node, nodes.Call):
        return TemplateError(
            "A template can't call anything; use a filter, such as {{ items | and_list }}."
        )
    if isinstance(node, nodes.BinExpr):
        return TemplateError(
            "A template can't do arithmetic; use a filter, such as {{ proportion | percent }}."
        )
    if isinstance(node, nodes.Stmt):
        return TemplateError(f"A template can't use {{% {_STATEMENT_KEYWORDS.get(name, name)} %}}.")
    return TemplateError(f"A template can't use {name} expressions.")


def _validate_loop(node: nodes.For, scope: Mapping[str, Shape]) -> None:
    _validate_node(node.iter, scope)
    if not isinstance(node.target, nodes.Name):
        raise TemplateError("A loop goes over one name at a time.")
    loop_scope = {
        **scope,
        "loop": LOOP_VARIABLES,
        node.target.name: _item_shape(_shape_of(node.iter, scope)),
    }
    for child in [*node.body, *node.else_, *([node.test] if node.test else [])]:
        _validate_node(child, loop_scope)


def _item_shape(shape: Shape) -> Shape:
    return shape.item if isinstance(shape, ListOf) else "any"


def _shape_of(node: nodes.Node, scope: Mapping[str, Shape]) -> Shape:
    if isinstance(node, nodes.Name):
        if node.name not in scope:
            raise TemplateError(f"There's no {node.name}.")
        return scope[node.name]
    if isinstance(node, nodes.Getattr | nodes.Getitem):
        return _member_shape(node, _shape_of(node.node, scope))
    if isinstance(node, nodes.Filter):
        return _filtered_shape(node, _shape_of(node.node, scope) if node.node else "any")
    return "any"


def unknown_step_message(step_id: str) -> str:
    """The error for a template reading a step that isn't before it."""
    return f"No step before this one is called {step_id}."


def _member_shape(node: nodes.Getattr | nodes.Getitem, owner_shape: Shape) -> Shape:
    if isinstance(node, nodes.Getitem):
        if not isinstance(node.arg, nodes.Const):
            if isinstance(owner_shape, Mapping):
                raise TemplateError("Name the field to read with a literal key.")
            return _item_shape(owner_shape)
        attribute = node.arg.value
        if isinstance(owner_shape, ListOf):
            return owner_shape.item
        if owner_shape == "text" and isinstance(attribute, int):
            return "text"
    else:
        attribute = node.attr
    if isinstance(owner_shape, Mapping):
        if attribute not in owner_shape:
            if _dotted_name(node.node) == "steps":
                raise TemplateError(unknown_step_message(str(attribute)))
            raise TemplateError(f"{_dotted_name(node.node)} has no {attribute}.")
        return owner_shape[attribute]
    if owner_shape == "any":
        return "any"
    raise TemplateError(f"{_dotted_name(node.node)} is {owner_shape}, which has no {attribute}.")


def _filtered_shape(node: nodes.Filter, shape: Shape) -> Shape:
    """The shape a filter gives of a value of ``shape``."""
    if node.name in ("selectattr", "rejectattr", "quoted", "code", "escaped", "list"):
        return shape
    if node.name == "map":
        attribute = next(
            (
                keyword.value.value
                for keyword in node.kwargs
                if keyword.key == "attribute" and isinstance(keyword.value, nodes.Const)
            ),
            None,
        )
        item_shape = _item_shape(shape)
        return ListOf(
            item_shape.get(attribute, "any")
            if attribute and isinstance(item_shape, Mapping)
            else "any"
        )
    if node.name in ("first", "last"):
        return _item_shape(shape)
    return "number" if node.name in ("length", "round") else "text"


def _dotted_name(node: nodes.Node) -> str:
    if isinstance(node, nodes.Getattr):
        return f"{_dotted_name(node.node)}.{node.attr}"
    return getattr(node, "name", "it")


def template_errors(
    step_data: Mapping[str, Any], scope: Mapping[str, Shape]
) -> Iterator[tuple[tuple[str, ...], TemplateError]]:
    """What isn't allowed in the condition and feedback of ``step_data``, where it is, when
    ``scope`` holds what they may name. Feedback may be a template for each outcome."""
    if isinstance(template := step_data.get("if"), str):
        try:
            validate(
                expression := condition_expression(template), scope, expression_kind="condition"
            )
        except TemplateError as error:
            yield ("if",), error
        else:
            if not top_level_names(expression):
                yield (
                    ("if",),
                    TemplateError(
                        "Make a condition read an earlier step or a setting, such as "
                        "{{ steps.test.output.correct }}."
                    ),
                )
    feedback = step_data.get("feedback")
    templates = feedback.items() if isinstance(feedback, Mapping) else [(None, feedback)]
    for outcome_key, template in templates:
        if isinstance(template, str):
            try:
                validate(template, {**scope, **FEEDBACK_SCOPE})
            except TemplateError as error:
                yield ("feedback",) if outcome_key is None else ("feedback", outcome_key), error


@lru_cache(maxsize=4096)
def names_read(template: str, root: str) -> frozenset[str]:
    """The names a template reads of ``root``, such as the steps a condition reads; none if it
    doesn't parse, which checking it reports."""
    try:
        tree = ENVIRONMENT.parse(template)
    except TemplateSyntaxError:
        return frozenset()
    attribute_names = {
        node.attr
        for node in tree.find_all(nodes.Getattr)
        if isinstance(node.node, nodes.Name) and node.node.name == root
    }
    indexed_names = {
        node.arg.value
        for node in tree.find_all(nodes.Getitem)
        if isinstance(node.node, nodes.Name)
        and node.node.name == root
        and isinstance(node.arg, nodes.Const)
        and isinstance(node.arg.value, str)
    }
    return frozenset(attribute_names | indexed_names)


def names_read_in(value: Any, root: str) -> frozenset[str]:
    """The names the templates anywhere in ``value`` read of ``root``."""
    return frozenset().union(*(names_read(text, root) for text in _strings(value)))


def render(template: str, variables: Mapping[str, Any]) -> str:
    return _template(template).render(**variables).strip()


def evaluate(expression: str, variables: Mapping[str, Any]) -> Any:
    return _expression(expression)(**variables)


# Marking fills the same templates in for every student: each is compiled once, not per student.
@lru_cache(maxsize=4096)
def _template(template: str) -> Template:
    return ENVIRONMENT.from_string(template)


@lru_cache(maxsize=4096)
def _expression(expression: str) -> TemplateExpression:
    return ENVIRONMENT.compile_expression(expression, undefined_to_none=False)
