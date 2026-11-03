"""A check's condition as rows, each testing an earlier check, of which all or any must hold."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

from jinja2 import TemplateSyntaxError, nodes

from ..methods.templates import ENVIRONMENT, reference

type Match = Literal["all", "any"]
type ConditionTest = Literal["correct", "incorrect", "half_correct"]


@dataclass(frozen=True)
class ConditionRow:
    check: str
    """The check it tests: in a ``MarkingDraft``, its path."""
    test: ConditionTest


@dataclass(frozen=True)
class Condition:
    rows: tuple[ConditionRow, ...] = ()
    match: Match = "all"

    test_labels: ClassVar[Mapping[ConditionTest, str]] = {
        "correct": "is correct",
        "incorrect": "isn't correct",
        "half_correct": "is at least half correct",
    }
    """What a row may test of its check, with how it reads."""
    match_labels: ClassVar[Mapping[Match, str]] = {"all": "All of these", "any": "Any of these"}
    """Whether all or any of the rows must hold, with how each reads."""

    def describe(self, check_label: Callable[[str], str] = str) -> str:
        """The condition in words, naming each check by ``check_label`` of what its row has:
        ``Only when Test is correct and Units isn't correct``."""
        conjunction = " and " if self.match == "all" else " or "
        return "Only when " + conjunction.join(
            f"{check_label(row.check)} {self.test_labels[row.test]}" for row in self.rows
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
    match: Match = "any" if isinstance(node, nodes.Or) else "all"
    rows = [
        _condition_row(operand)
        for operand in _operands(node, nodes.Or if match == "any" else nodes.And)
    ]
    valid_rows = [row for row in rows if row is not None]
    return Condition(tuple(valid_rows), match) if len(valid_rows) == len(rows) else None


def condition_template(condition: Condition) -> str:
    """The condition as a check's ``if``, its rows naming checks by ID; empty, so always, without
    a row."""
    parts = [_row_template(row) for row in condition.rows if row.check]
    expression = f" {'or' if condition.match == 'any' else 'and'} ".join(parts)
    return f"{{{{ {expression} }}}}" if parts else ""


def _row_template(row: ConditionRow) -> str:
    match row.test:
        case "incorrect":
            return f"not checks.{row.check}.correct"
        case "half_correct":
            return f"checks.{row.check}.proportion >= 0.5"
    return f"checks.{row.check}.correct"


def _operands(node: nodes.Node, operator: type[nodes.BinExpr]) -> list[nodes.Node]:
    if isinstance(node, operator):
        return [*_operands(node.left, operator), *_operands(node.right, operator)]
    return [node]


def _condition_row(node: nodes.Node) -> ConditionRow | None:
    if isinstance(node, nodes.Not):
        check = _check_id_read(node.node, "correct")
        return ConditionRow(check, "incorrect") if check else None
    if isinstance(node, nodes.Compare):
        (operand,) = node.ops if len(node.ops) == 1 else (None,)
        is_half_test = (
            operand is not None and operand.op == "gteq" and _is_constant(operand.expr, 0.5)
        )
        check = _check_id_read(node.expr, "proportion") if is_half_test else None
        return ConditionRow(check, "half_correct") if check else None
    check = _check_id_read(node, "correct")
    return ConditionRow(check, "correct") if check else None


def _is_constant(node: nodes.Node, value: object) -> bool:
    return isinstance(node, nodes.Const) and node.value == value


def _check_id_read(node: nodes.Node, attribute: str) -> str | None:
    """The ID of the check whose result ``node`` reads ``attribute`` of, as in
    ``checks.<id>.correct``."""
    if not (isinstance(node, nodes.Getattr) and node.attr == attribute):
        return None
    owner = node.node
    if isinstance(owner, nodes.Getattr) and isinstance(owner.node, nodes.Name):
        return owner.attr if owner.node.name == "checks" else None
    return None
