"""Sandboxed Jinja for references, conditions and templates, with an allowlist checked on load."""

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Literal

from jinja2 import StrictUndefined, Template, TemplateSyntaxError, meta, nodes
from jinja2.environment import TemplateExpression
from jinja2.sandbox import SandboxedEnvironment

from ..common.exceptions import GradeFlowError
from ..common.text import (
    format_number,
    markdown_code,
    markdown_list,
    plain_join,
    plain_markdown,
    quoted,
)

REFERENCE = re.compile(r"^\{\{((?:(?!\{\{|\}\}).)+)\}\}$", re.DOTALL)
"""A value that is exactly one reference, which keeps the type of what it refers to."""


def _each(write: Callable[[Any], str]) -> Callable[[Any], Any]:
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
_OWN_FILTERS: dict[str, Callable[..., Any]] = {
    "and_list": partial(plain_join, conjunction="and"),
    "or_list": partial(plain_join, conjunction="or"),
    "quoted": _each(quoted),
    "code": _each(markdown_code),
    "percent": lambda value: f"{value:.0%}",
    "described": lambda checks: markdown_list(check.headed() for check in checks),
    "description": lambda check: plain_markdown(check.describe()),
}
ENVIRONMENT.filters.update(_OWN_FILTERS)
FILTERS = frozenset(
    {
        *_OWN_FILTERS,
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
_ALLOWED = (
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

_KEYWORDS = {
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


CHECK_RESULT: Mapping[str, Shape] = {
    "correct": "bool",
    "proportion": "number",
    "feedback": "text",
    "marked": "bool",
    "skipped": "bool",
}
"""An earlier check's result, as ``checks.<id>``."""
PART: Mapping[str, Shape] = {**CHECK_RESULT, "name": "text", "question": "text"}
"""One of a check's own checks' results, as an item of ``parts``."""
LOOP: Mapping[str, Shape] = {"index": "number", "first": "bool", "last": "bool", "length": "number"}
MARKED: Mapping[str, Shape] = {
    "answer": "text",
    "correct": "bool",
    "proportion": "number",
    "feedback": "text",
    "parts": ListOf(PART),
}
"""What a check's feedback template sees of how it marked: `feedback` is its method's own."""


class TemplateError(GradeFlowError, ValueError):
    """A template, condition or reference that isn't allowed, or names something not in scope."""


def reference(value: Any) -> str | None:
    """The expression of a value that is exactly one ``{{ … }}``."""
    if isinstance(value, str) and (found := REFERENCE.fullmatch(value.strip())):
        return found[1].strip()
    return None


def condition(text: str) -> str:
    """The expression of an ``if``, which must be exactly one ``{{ … }}``."""
    if (expression := reference(text)) is None:
        raise TemplateError("Write a condition as one {{ … }}, such as {{ checks.test.correct }}.")
    return expression


@lru_cache(maxsize=4096)
def roots(expression: str) -> frozenset[str]:
    """The names ``expression`` reads, such as ``inputs`` and ``checks``."""
    return frozenset(meta.find_undeclared_variables(ENVIRONMENT.parse(f"{{{{ {expression} }}}}")))


def is_template(text: str) -> bool:
    """Whether ``text`` has a reference or a statement in it, so it is filled in when it marks."""
    return "{{" in text or "{%" in text


def templated(value: Any) -> bool:
    """Whether ``value`` has a reference or template in it, so it is only known when it marks."""
    if isinstance(value, str):
        return is_template(value)
    if isinstance(value, Mapping):
        return any(templated(item) for item in value.values())
    if isinstance(value, list):
        return any(templated(item) for item in value)
    return False


def validate(source: str, scope: Mapping[str, Shape], *, expression: bool = False) -> None:
    """Refuses a template, or an expression, that uses anything not allowed or not in
    ``scope``, the types of what it may name."""
    try:
        tree = ENVIRONMENT.parse(f"{{{{ {source} }}}}" if expression else source)
    except TemplateSyntaxError as error:
        if expression:
            raise TemplateError("Can't read this condition.") from None
        raise TemplateError(f"{error.message} (line {error.lineno})") from None
    _walk(tree, dict(scope))


def _walk(node: nodes.Node, scope: dict[str, Shape]) -> None:
    if not isinstance(node, _ALLOWED):
        raise _refusal(node)
    if isinstance(node, nodes.Getattr) and node.attr.startswith("_"):
        raise TemplateError(f"There's no {node.attr}.")
    if isinstance(node, nodes.For):
        _walk_loop(node, scope)
        return
    if isinstance(node, nodes.Filter) and node.name not in FILTERS:
        raise TemplateError(f"There's no filter {node.name}.")
    if isinstance(node, nodes.Name | nodes.Getattr | nodes.Getitem):
        _type(node, scope)
    for child in node.iter_child_nodes():
        _walk(child, scope)


def _refusal(node: nodes.Node) -> TemplateError:
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
        return TemplateError(f"A template can't use {{% {_KEYWORDS.get(name, name)} %}}.")
    return TemplateError(f"A template can't use {name} expressions.")


def _walk_loop(node: nodes.For, scope: dict[str, Shape]) -> None:
    _walk(node.iter, scope)
    if not isinstance(node.target, nodes.Name):
        raise TemplateError("A loop goes over one name at a time.")
    inner = {**scope, "loop": LOOP, node.target.name: _element(_type(node.iter, scope))}
    for child in [*node.body, *node.else_, *([node.test] if node.test else [])]:
        _walk(child, inner)


def _element(kind: Shape) -> Shape:
    return kind.item if isinstance(kind, ListOf) else "any"


def _type(node: nodes.Node, scope: Mapping[str, Shape]) -> Shape:
    if isinstance(node, nodes.Name):
        if node.name not in scope:
            raise TemplateError(f"There's no {node.name}.")
        return scope[node.name]
    if isinstance(node, nodes.Getattr | nodes.Getitem):
        return _member(node, _type(node.node, scope))
    if isinstance(node, nodes.Filter):
        return _filtered(node, _type(node.node, scope) if node.node else "any")
    return "any"


def unknown_check(name: str) -> str:
    """What a condition reading a check that isn't before it is told."""
    return f"No check before this one is called {name}."


def _member(node: nodes.Getattr | nodes.Getitem, owner: Shape) -> Shape:
    if isinstance(node, nodes.Getitem):
        if not isinstance(node.arg, nodes.Const):
            if isinstance(owner, Mapping):
                raise TemplateError("Name the field to read with a literal key.")
            return _element(owner)
        attribute = node.arg.value
        if isinstance(owner, ListOf):
            return owner.item
        if owner == "text" and isinstance(attribute, int):
            return "text"
    else:
        attribute = node.attr
    if isinstance(owner, Mapping):
        if attribute not in owner:
            if _path(node.node) == "checks":
                raise TemplateError(unknown_check(str(attribute)))
            raise TemplateError(f"{_path(node.node)} has no {attribute}.")
        return owner[attribute]
    if owner == "any":
        return "any"
    raise TemplateError(f"{_path(node.node)} is {owner}, which has no {attribute}.")


def _filtered(node: nodes.Filter, inner: Shape) -> Shape:
    """What a filter gives of a value of type ``inner``."""
    if node.name in ("selectattr", "rejectattr", "quoted", "code", "list"):
        return inner
    if node.name == "map":
        attribute = next(
            (
                k.value.value
                for k in node.kwargs
                if k.key == "attribute" and isinstance(k.value, nodes.Const)
            ),
            None,
        )
        fields = _element(inner)
        return ListOf(
            fields.get(attribute, "any") if attribute and isinstance(fields, Mapping) else "any"
        )
    if node.name in ("first", "last"):
        return _element(inner)
    return "number" if node.name in ("length", "round") else "text"


def _path(node: nodes.Node) -> str:
    if isinstance(node, nodes.Getattr):
        return f"{_path(node.node)}.{node.attr}"
    return getattr(node, "name", "it")


def template_errors(
    check: Mapping[str, Any], scope: Mapping[str, Shape]
) -> Iterator[tuple[tuple[str, ...], TemplateError]]:
    """What isn't allowed in the condition and feedback of ``check``, where it is, when ``scope``
    holds what they may name. Feedback may be a template for each outcome."""
    if isinstance(source := check.get("if"), str):
        try:
            validate(expression := condition(source), scope, expression=True)
        except TemplateError as error:
            yield ("if",), error
        else:
            if not roots(expression):
                yield (
                    ("if",),
                    TemplateError(
                        "Make a condition read an earlier check or an input, such as "
                        "{{ checks.test.correct }}."
                    ),
                )
    feedback = check.get("feedback")
    sources = feedback.items() if isinstance(feedback, Mapping) else [(None, feedback)]
    for outcome, source in sources:
        if isinstance(source, str):
            try:
                validate(source, {**scope, **MARKED})
            except TemplateError as error:
                yield ("feedback",) if outcome is None else ("feedback", outcome), error


@lru_cache(maxsize=4096)
def named(source: str, root: str) -> frozenset[str]:
    """The names a template reads of ``root``, such as the checks a condition reads; none if it
    doesn't parse, which checking it reports."""
    try:
        tree = ENVIRONMENT.parse(source)
    except TemplateSyntaxError:
        return frozenset()
    attributes = {
        node.attr
        for node in tree.find_all(nodes.Getattr)
        if isinstance(node.node, nodes.Name) and node.node.name == root
    }
    indexed = {
        node.arg.value
        for node in tree.find_all(nodes.Getitem)
        if isinstance(node.node, nodes.Name)
        and node.node.name == root
        and isinstance(node.arg, nodes.Const)
        and isinstance(node.arg.value, str)
    }
    return frozenset(attributes | indexed)


def named_in(value: Any, root: str) -> frozenset[str]:
    """The names the templates anywhere in ``value`` read of ``root``."""
    if isinstance(value, Mapping):
        return frozenset().union(*(named_in(item, root) for item in value.values()))
    if isinstance(value, list):
        return frozenset().union(*(named_in(item, root) for item in value))
    return named(value, root) if isinstance(value, str) else frozenset()


def render(source: str, context: Mapping[str, Any]) -> str:
    return _template(source).render(**context).strip()


def evaluate(expression: str, context: Mapping[str, Any]) -> Any:
    return _expression(expression)(**context)


# Marking fills the same templates in for every student: each is compiled once, not per student.
@lru_cache(maxsize=4096)
def _template(source: str) -> Template:
    return ENVIRONMENT.from_string(source)


@lru_cache(maxsize=4096)
def _expression(expression: str) -> TemplateExpression:
    return ENVIRONMENT.compile_expression(expression, undefined_to_none=False)
