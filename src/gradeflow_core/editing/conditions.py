"""A check's condition as rows, each testing an earlier check, of which all or any must hold."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

from jinja2 import TemplateSyntaxError, nodes

from ..methods.templates import ENVIRONMENT, reference

type Join = Literal["all", "any"]
type ConditionTest = Literal["correct", "incorrect", "half"]


@dataclass(frozen=True)
class ConditionRow:
    check: str
    """The check it tests: in a ``MarkingDraft``, its path."""
    test: ConditionTest


@dataclass(frozen=True)
class Condition:
    rows: tuple[ConditionRow, ...] = ()
    join: Join = "all"

    tests: ClassVar[Mapping[ConditionTest, str]] = {
        "correct": "is correct",
        "incorrect": "isn't correct",
        "half": "is at least half correct",
    }
    """What a row may test of its check, with how it reads."""
    joins: ClassVar[Mapping[Join, str]] = {"all": "All of these", "any": "Any of these"}
    """Whether all or any of the rows must hold, with how each reads."""

    def describe(self, label: Callable[[str], str] = str) -> str:
        """The condition in words, naming each check by ``label`` of what its row has:
        ``Only when Test is correct and Units isn't correct``."""
        joined = " and " if self.join == "all" else " or "
        return "Only when " + joined.join(
            f"{label(row.check)} {self.tests[row.test]}" for row in self.rows
        )


def read_condition(text: str) -> Condition | None:
    """The rows of a condition made of them, each naming its check by ID; None for any other."""
    expression = reference(text)
    try:
        tree = ENVIRONMENT.parse(f"{{{{ {expression} }}}}") if expression else None
    except TemplateSyntaxError:
        return None
    if tree is None or len(tree.body) != 1 or not isinstance(output := tree.body[0], nodes.Output):
        return None
    node = output.nodes[0]
    join: Join = "any" if isinstance(node, nodes.Or) else "all"
    rows = [_row(term) for term in _terms(node, nodes.Or if join == "any" else nodes.And)]
    found = [row for row in rows if row is not None]
    return Condition(tuple(found), join) if len(found) == len(rows) else None


def written_condition(condition: Condition) -> str:
    """The condition as a check's ``if``, its rows naming checks by ID; empty, so always, without
    a row."""
    parts = [_written(row) for row in condition.rows if row.check]
    joined = f" {'or' if condition.join == 'any' else 'and'} ".join(parts)
    return f"{{{{ {joined} }}}}" if parts else ""


def _written(row: ConditionRow) -> str:
    match row.test:
        case "incorrect":
            return f"not checks.{row.check}.correct"
        case "half":
            return f"checks.{row.check}.proportion >= 0.5"
    return f"checks.{row.check}.correct"


def _terms(node: nodes.Node, kind: type[nodes.BinExpr]) -> list[nodes.Node]:
    if isinstance(node, kind):
        return [*_terms(node.left, kind), *_terms(node.right, kind)]
    return [node]


def _row(node: nodes.Node) -> ConditionRow | None:
    if isinstance(node, nodes.Not):
        check = _result(node.node, "correct")
        return ConditionRow(check, "incorrect") if check else None
    if isinstance(node, nodes.Compare):
        (operand,) = node.ops if len(node.ops) == 1 else (None,)
        half = operand is not None and operand.op == "gteq" and _is(operand.expr, 0.5)
        check = _result(node.expr, "proportion") if half else None
        return ConditionRow(check, "half") if check else None
    check = _result(node, "correct")
    return ConditionRow(check, "correct") if check else None


def _is(node: nodes.Node, value: object) -> bool:
    return isinstance(node, nodes.Const) and node.value == value


def _result(node: nodes.Node, attribute: str) -> str | None:
    """The ID of the check whose result ``node`` reads ``attribute`` of, as in
    ``checks.<id>.correct``."""
    if not (isinstance(node, nodes.Getattr) and node.attr == attribute):
        return None
    owner = node.node
    if isinstance(owner, nodes.Getattr) and isinstance(owner.node, nodes.Name):
        return owner.attr if owner.node.name == "checks" else None
    return None
