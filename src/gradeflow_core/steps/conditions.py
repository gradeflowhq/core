"""A step's condition as rows, each testing an earlier check or a setting, of which all or any
must hold."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

from jinja2 import nodes

from .templates import expression_node

type Match = Literal["all", "any"]
type CheckTest = Literal["correct", "incorrect", "half_correct"]

CHECK_TESTS: Mapping[CheckTest, str] = {
    "correct": "is correct",
    "incorrect": "isn't correct",
    "half_correct": "is at least half correct",
}
"""What a row may test of a check, with its words."""
YES_NO_TESTS: Mapping[bool, str] = {True: "is on", False: "is off"}
"""What a row may test of a Yes/No setting, with its words."""


@dataclass(frozen=True)
class CheckRow:
    check: str
    """The check it tests: in a draft, its path."""
    test: CheckTest


@dataclass(frozen=True)
class SettingRow:
    setting: str
    """The setting it tests, by name."""
    value: bool | str | float
    """True or False for a Yes/No setting, else one of its choices."""


type ConditionRow = CheckRow | SettingRow


@dataclass(frozen=True)
class ConditionSubject:
    """A check or setting a condition may test, with the words of each row testing it."""

    label: str
    rows: Mapping[ConditionRow, str]


def check_subject(check: str, label: str) -> ConditionSubject:
    """The check ``check`` as a condition may test it."""
    return ConditionSubject(
        label, {CheckRow(check, test): words for test, words in CHECK_TESTS.items()}
    )


@dataclass(frozen=True)
class Condition:
    rows: tuple[ConditionRow, ...] = ()
    match: Match = "all"

    match_labels: ClassVar[Mapping[Match, str]] = {"all": "All of these", "any": "Any of these"}
    """Whether all or any of the rows must hold, with how each reads."""

    def describe(self, subjects: Iterable[ConditionSubject]) -> str | None:
        """The condition in words: ``Only when Test is correct and Strict is on``. None if a row
        tests none of ``subjects``."""
        words = {
            row: f"{subject.label} {test}"
            for subject in subjects
            for row, test in subject.rows.items()
        }
        if any(row not in words for row in self.rows):
            return None
        conjunction = " and " if self.match == "all" else " or "
        return "Only when " + conjunction.join(words[row] for row in self.rows)


def read_condition(template: str) -> Condition | None:
    """The rows of a condition made of them, each naming its check by ID; None for any other."""
    node = expression_node(template)
    if node is None:
        return None
    match: Match = "any" if isinstance(node, nodes.Or) else "all"
    rows = [
        _condition_row(operand)
        for operand in _operands(node, nodes.Or if match == "any" else nodes.And)
    ]
    valid_rows = [row for row in rows if row is not None]
    return Condition(tuple(valid_rows), match) if len(valid_rows) == len(rows) else None


def condition_note(template: str, subjects: Iterable[ConditionSubject]) -> str:
    """The condition ``template`` in words, with ``subjects`` naming checks by ID."""
    condition = read_condition(template)
    return (condition and condition.describe(subjects)) or "Only when its condition holds"


def condition_template(condition: Condition) -> str:
    """The condition as a step's ``if``, naming checks by ID; empty without a row."""
    row_templates = [_row_template(row) for row in condition.rows]
    expression = f" {'or' if condition.match == 'any' else 'and'} ".join(row_templates)
    return f"{{{{ {expression} }}}}" if row_templates else ""


def _row_template(row: ConditionRow) -> str:
    match row:
        case CheckRow(check, "incorrect"):
            return f"not steps.{check}.output.correct"
        case CheckRow(check, "half_correct"):
            return f"steps.{check}.output.proportion >= 0.5"
        case CheckRow(check):
            return f"steps.{check}.output.correct"
        case SettingRow(setting, True):
            return f"settings.{setting}"
        case SettingRow(setting, False):
            return f"not settings.{setting}"
        case SettingRow(setting, value):
            return f"settings.{setting} == {value!r}"


def _operands(node: nodes.Node, operator: type[nodes.BinExpr]) -> list[nodes.Node]:
    if isinstance(node, operator):
        return [*_operands(node.left, operator), *_operands(node.right, operator)]
    return [node]


def _condition_row(node: nodes.Node) -> ConditionRow | None:
    match node:
        case nodes.Not(node=operand):
            if check := _check_id_read(operand, "correct"):
                return CheckRow(check, "incorrect")
            setting = _setting_read(operand)
            return SettingRow(setting, False) if setting else None
        case nodes.Compare(
            expr=operand, ops=[nodes.Operand(op="gteq", expr=nodes.Const(value=0.5))]
        ):
            check = _check_id_read(operand, "proportion")
            return CheckRow(check, "half_correct") if check else None
        case nodes.Compare(
            expr=operand,
            ops=[nodes.Operand(op="eq", expr=nodes.Const(value=str() | int() | float() as value))],
        ) if not isinstance(value, bool):
            setting = _setting_read(operand)
            return SettingRow(setting, value) if setting else None
    if check := _check_id_read(node, "correct"):
        return CheckRow(check, "correct")
    setting = _setting_read(node)
    return SettingRow(setting, True) if setting else None


def _check_id_read(node: nodes.Node, attribute: str) -> str | None:
    """The ID of the check whose result ``node`` reads ``attribute`` of, as in
    ``steps.<id>.output.correct``."""
    match node:
        case nodes.Getattr(
            node=nodes.Getattr(
                node=nodes.Getattr(node=nodes.Name(name="steps"), attr=check_id), attr="output"
            ),
            attr=attribute_read,
        ):
            return check_id if attribute_read == attribute else None
    return None


def _setting_read(node: nodes.Node) -> str | None:
    """The setting ``node`` reads, as in ``settings.<name>``."""
    match node:
        case nodes.Getattr(node=nodes.Name(name="settings"), attr=name):
            return name
    return None
