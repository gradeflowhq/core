from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import Field

from ...common.hints import ADVANCED_HINT, PYTHON_CODE_HINT, hint
from ...common.problems import Problem
from ...common.text import as_sentence, as_text, code_block, quoted
from ...questions.base import Question
from ...questions.types import ALL_TYPES, Answer
from ..check import Check
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import Outcome
from ..position import Position
from ..settings import Settings, specialized
from . import runner
from .parameters import PARAMETERS_FIELD, Parameter, ParameterName, parameter_values

RESULT_VARIABLES_HELP = (
    "correct (True or False) or proportion (from 0 to 1), and feedback if you like"
)
STARTER_CODE = 'correct = False\nfeedback = ""\n'
TIME_LIMIT_FIELD = Field(
    default=runner.TIME_LIMIT_S,
    ge=1,
    le=runner.MAX_TIME_LIMIT_S,
    title="Time limit (seconds)",
    description="Longest time the code may run before it fails.",
    json_schema_extra=ADVANCED_HINT,
)
MEMORY_LIMIT_FIELD = Field(
    default=runner.MEMORY_LIMIT_MB,
    ge=runner.MIN_MEMORY_LIMIT_MB,
    le=runner.MAX_MEMORY_LIMIT_MB,
    title="Memory limit (MiB)",
    description="Most memory the code may use before it fails.",
    json_schema_extra=ADVANCED_HINT,
)


def code_description(answer: str) -> str:
    return f"Reads {answer} and any parameters, and sets {RESULT_VARIABLES_HELP}."


class CustomCodeSettings(Settings):
    code: str = Field(
        default=STARTER_CODE,
        min_length=1,
        title="Code",
        description=code_description("answer"),
        json_schema_extra=hint(**PYTHON_CODE_HINT),
    )
    parameters: dict[ParameterName, Parameter] = PARAMETERS_FIELD
    time_limit: int = TIME_LIMIT_FIELD
    memory_limit: int = MEMORY_LIMIT_FIELD


def outcome_from_variables(variables: Mapping[str, Any], question: str = "") -> Outcome:
    """The marks code set: ``proportion`` from 0 to 1, or else ``correct``, with its
    ``feedback``."""
    feedback = str(variables.get("feedback", ""))
    proportion = variables.get("proportion")
    if proportion is None:
        return Outcome(float(bool(variables.get("correct", False))), feedback)
    name = f"proportion for {question}" if question else "proportion"
    if (
        isinstance(proportion, bool)
        or not isinstance(proportion, int | float)
        or not 0 <= proportion <= 1
    ):
        shown = quoted(proportion) if isinstance(proportion, str) else as_text(proportion)
        return code_failure(f"Your code set {name} to {shown}. It must be a number from 0 to 1.")
    return Outcome(float(proportion), feedback)


def code_failure(reason: str) -> Outcome:
    """The marker's code failed, not the student's answer: a person marks it."""
    return Outcome(0.0, reason, marked=False)


def run_failure(
    code: str, variables: dict[str, Any], time_limit_s: int, memory_limit_mb: int
) -> str | None:
    """Runs ``code``, which may change ``variables``; why it failed, if it did."""
    try:
        runner.run(code, variables, time_limit_s=time_limit_s, memory_limit_mb=memory_limit_mb)
    except runner.CodeError as error:
        return as_sentence(f"Your code failed: {error.reason}")
    return None


def code_problems(code: str) -> Iterator[Problem]:
    """Code that can't run at all, at the line at fault."""
    try:
        compile(code, "<code>", "exec")
    except SyntaxError as error:
        line_note = f"Line {error.lineno}: " if error.lineno else ""
        yield Problem(as_sentence(f"{line_note}{error.msg}"), field=("code",))


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
    settings = CustomCodeSettings
    examples = examples_beside(__file__)

    def settings_model_at(self, position: Position) -> type[Settings]:
        question = position.question_model
        answer = f"answer ({question.python_type()})" if question else "answer"
        return specialized(self.settings, code={"description": code_description(answer)})

    def describe(self, check: Check) -> str:
        return f"Python code decides the marks:\n\n{code_block(self._settings_of(check).code)}"

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        return code_problems(self._settings_of(check).code)

    def _mark_answer(
        self, settings: CustomCodeSettings, answer: Answer, question: Question
    ) -> Outcome:
        variables: dict[str, Any] = {"answer": answer, **parameter_values(settings.parameters)}
        if (
            error := run_failure(
                settings.code, variables, settings.time_limit, settings.memory_limit
            )
        ) is not None:
            return code_failure(error)
        return outcome_from_variables(variables)
