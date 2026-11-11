"""What a step's ``input`` names, in place of the answer as it is."""

from dataclasses import dataclass
from functools import lru_cache

from jinja2 import TemplateError as JinjaError
from jinja2 import nodes

from ..questions.types import QuestionId
from .templates import evaluate, expression_node, reference_expression

ANSWER_INPUT = "{{ answer }}"
"""What a step without an ``input`` reads."""
_INPUT_RULE = "Write an input as one of {{ steps.name.output }}, {{ values[0] }}, {{ answers.Q1 }}."


@dataclass(frozen=True)
class AnswerSource:
    """The answer as it is."""


@dataclass(frozen=True)
class AnswerValueSource:
    """One value of the answer."""

    value_number: int
    """From 1."""


@dataclass(frozen=True)
class OtherAnswerSource:
    """The answer to another question."""

    question_id: QuestionId


@dataclass(frozen=True)
class StepOutputSource:
    """What an earlier step gives."""

    step_id: str


InputSource = AnswerSource | AnswerValueSource | OtherAnswerSource | StepOutputSource


class _Values:
    def __getitem__(self, index: object) -> AnswerValueSource:
        if type(index) is not int or index < 0:
            raise LookupError(index)
        return AnswerValueSource(index + 1)


class _Answers:
    def __getitem__(self, question_id: object) -> OtherAnswerSource:
        if not isinstance(question_id, str):
            raise LookupError(question_id)
        return OtherAnswerSource(question_id)


@dataclass(frozen=True)
class _Step:
    output: StepOutputSource


class _Steps:
    def __getitem__(self, step_id: object) -> _Step:
        if not isinstance(step_id, str):
            raise LookupError(step_id)
        return _Step(StepOutputSource(step_id))


_NAMES = {"answer": AnswerSource(), "values": _Values(), "answers": _Answers(), "steps": _Steps()}
"""What an input may name, each standing for the source it reads."""
_ACCESS_NODES = (nodes.Name, nodes.Getattr, nodes.Getitem, nodes.Const)


@lru_cache(maxsize=4096)
def input_source(input_text: str | None) -> InputSource:
    """The source a step's ``input`` names. Raises ``ValueError`` if it names none."""
    text = ANSWER_INPUT if input_text is None else input_text
    expression, node = reference_expression(text), expression_node(text)
    if expression is None or node is None or not _only_reads_names(node):
        raise ValueError(_INPUT_RULE)
    try:
        source = evaluate(expression, _NAMES)
    except JinjaError:
        raise ValueError(_INPUT_RULE) from None
    if not isinstance(source, InputSource):
        raise ValueError(_INPUT_RULE)
    return source


def _only_reads_names(node: nodes.Node) -> bool:
    """Whether an expression is a name with attributes and items of it, and nothing computed."""
    return isinstance(node, _ACCESS_NODES) and all(
        _only_reads_names(child) for child in node.iter_child_nodes()
    )


def input_text(source: InputSource) -> str:
    """How a step's ``input`` names ``source``."""
    match source:
        case AnswerSource():
            return ANSWER_INPUT
        case AnswerValueSource(value_number=value_number):
            return f"{{{{ values[{value_number - 1}] }}}}"
        case OtherAnswerSource(question_id=question_id) if question_id.isidentifier():
            return f"{{{{ answers.{question_id} }}}}"
        case OtherAnswerSource(question_id=question_id):
            return f"{{{{ answers[{question_id!r}] }}}}"
        case StepOutputSource(step_id=step_id):
            return f"{{{{ steps.{step_id}.output }}}}"
