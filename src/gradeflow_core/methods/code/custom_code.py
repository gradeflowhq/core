from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import Field

from ...common.hints import CODE, hint
from ...common.problems import Problem
from ...common.text import as_text, fenced, quoted, sentence
from ...questions.base import Question
from ...questions.types import ALL_TYPES, Answer
from ..check import Check
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import Outcome
from ..position import Position
from ..settings import Settings, specialised
from . import runner
from .parameters import PARAMETERS, Parameter, ParameterName, values

MARKS = "correct (True or False) or proportion (from 0 to 1), and feedback if you like"
STARTER = 'correct = False\nfeedback = ""\n'


def code_help(answer: str) -> str:
    return f"Reads {answer} and any parameters, and sets {MARKS}."


class CustomCodeSettings(Settings):
    code: str = Field(
        default=STARTER,
        min_length=1,
        title="Code",
        description=code_help("answer"),
        json_schema_extra=hint(**CODE),
    )
    parameters: dict[ParameterName, Parameter] = PARAMETERS


def outcome_of(variables: Mapping[str, Any], subject: str = "") -> Outcome:
    """The marks code set: ``proportion`` from 0 to 1, or else ``correct``, with its
    ``feedback``."""
    feedback = str(variables.get("feedback", ""))
    proportion = variables.get("proportion")
    if proportion is None:
        return Outcome(float(bool(variables.get("correct", False))), feedback)
    name = f"proportion for {subject}" if subject else "proportion"
    if (
        isinstance(proportion, bool)
        or not isinstance(proportion, int | float)
        or not 0 <= proportion <= 1
    ):
        shown = quoted(proportion) if isinstance(proportion, str) else as_text(proportion)
        return failed(f"Your code set {name} to {shown}. It must be a number from 0 to 1.")
    return Outcome(float(proportion), feedback)


def failed(reason: str) -> Outcome:
    """The marker's code failed, not the student's answer: a person marks it."""
    return Outcome(0.0, reason, marked=False)


def run(code: str, variables: dict[str, Any]) -> str | None:
    """Runs ``code``, which may change ``variables``; why it failed, if it did."""
    try:
        runner.run(code, variables, time_limit_s=runner.TIME_LIMIT_S)
    except runner.CodeError as error:
        return sentence(f"Your code failed: {error.reason}")
    return None


def code_problems(code: str) -> Iterator[Problem]:
    """Code that can't run at all, at the line at fault."""
    try:
        compile(code, "<code>", "exec")
    except SyntaxError as error:
        where = f"Line {error.lineno}: " if error.lineno else ""
        yield Problem(sentence(f"{where}{error.msg}"), field=("code",))


class CustomCode(AnswerMethod[CustomCodeSettings]):
    name = "gradeflow/custom-code"
    version = 1
    label = "Custom Code"
    summary = "Your Python code decides the marks."
    guide = (
        "**Use it when:** No other method fits, and you can write Python.\n\n"
        "**How it marks:** Your code gets the student's answer as `answer`, and sets `correct` to "
        "True or False, or `proportion` to a number from 0 to 1. It may also set `feedback`, which "
        "students see as plain text.\n\n"
        "**Good to know:** Put values you may change in **Parameters**, so you can change them "
        "without editing the code.\n\n"
        "**Related:** To mark several questions together in code, use Custom Code for a Group."
    )
    level = "advanced"
    question_types = ALL_TYPES
    answer_shape = "both"
    settings = CustomCodeSettings
    examples = examples_beside(__file__)

    def settings_at(self, position: Position) -> type[Settings]:
        question = position.question_model
        answer = f"answer ({question.python_type()})" if question else "answer"
        return specialised(self.settings, code={"description": code_help(answer)})

    def describe(self, check: Check) -> str:
        return f"Python code decides the marks:\n\n{fenced(self._settings_of(check).code)}"

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        return code_problems(self._settings_of(check).code)

    def _mark_answer(
        self, settings: CustomCodeSettings, answer: Answer, question: Question
    ) -> Outcome:
        variables: dict[str, Any] = {"answer": answer, **values(settings.parameters)}
        if (error := run(settings.code, variables)) is not None:
            return failed(error)
        return outcome_of(variables)
