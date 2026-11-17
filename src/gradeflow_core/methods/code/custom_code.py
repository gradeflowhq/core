from collections.abc import Iterator, Mapping
from typing import Any, Literal

from pydantic import Field

from ...common.hints import ADVANCED_HINT, hint
from ...common.problems import Problem
from ...common.text import as_sentence, as_text, code_block, quoted
from ...questions.base import Question
from ...questions.types import ALL_QUESTION_TYPES, Answer, QuestionId
from ...steps.example import examples_beside
from ...steps.method import AnswerMethod
from ...steps.outcome import Outcome
from ...steps.position import Position
from ...steps.settings import Settings, specialized
from . import runner
from .parameters import PARAMETERS_FIELD, Parameter, ParameterName, parameter_values

RESULT_VARIABLES_HELP = (
    "`correct` (True or False) or `proportion` (from 0 to 1), and `feedback` if you like"
)
RESULT_VARIABLES = ("correct", "proportion", "feedback")
STARTER_CODE = 'correct = False\nfeedback = ""\n'


def time_limit_field(description: str | None = None) -> int:
    """A time limit setting, within the runner's bounds."""
    return Field(
        default=runner.TIME_LIMIT_S,
        ge=1,
        le=runner.MAX_TIME_LIMIT_S,
        title="Time limit (seconds)",
        description=description,
        json_schema_extra=ADVANCED_HINT,
    )


def memory_limit_field(description: str | None = None) -> int:
    """A memory limit setting, within the runner's bounds."""
    return Field(
        default=runner.MEMORY_LIMIT_MB,
        ge=runner.MIN_MEMORY_LIMIT_MB,
        le=runner.MAX_MEMORY_LIMIT_MB,
        title="Memory limit (MiB)",
        description=description,
        json_schema_extra=ADVANCED_HINT,
    )


TIME_LIMIT_FIELD = time_limit_field()
MEMORY_LIMIT_FIELD = memory_limit_field()


def code_description(answer_description: str) -> str:
    return (
        f"The answer is in {answer_description}, and each parameter in a variable of its name. "
        f"Set {RESULT_VARIABLES_HELP}."
    )


class CustomCodeSettings(Settings):
    code: str = Field(
        default=STARTER_CODE,
        min_length=1,
        title="Code",
        description=code_description("`answer`"),
        json_schema_extra=hint(kind="code", language="python"),
    )
    parameters: dict[ParameterName, Parameter] = PARAMETERS_FIELD
    time_limit: int = TIME_LIMIT_FIELD
    memory_limit: int = MEMORY_LIMIT_FIELD


def outcome_from_variables(variables: Mapping[str, Any], question_id: QuestionId = "") -> Outcome:
    """The marks code set: ``proportion`` from 0 to 1, or else ``correct``, with its
    ``feedback``."""
    feedback = "" if (value := variables.get("feedback")) is None else str(value)
    proportion = variables.get("proportion")
    if proportion is None:
        return Outcome(float(bool(variables.get("correct", False))), feedback)
    proportion_label = f"proportion for {question_id}" if question_id else "proportion"
    if (
        isinstance(proportion, bool)
        or not isinstance(proportion, int | float)
        or not 0 <= proportion <= 1
    ):
        proportion_text = quoted(proportion) if isinstance(proportion, str) else as_text(proportion)
        return code_failure_outcome(
            f"Your code set {proportion_label} to {proportion_text}. "
            "It must be a number from 0 to 1."
        )
    return Outcome(float(proportion), feedback)


def code_failure_outcome(reason: str) -> Outcome:
    """The marker's code failed, not the student's answer: a person marks it."""
    return Outcome(0.0, reason, marked=False)


def code_error_outcome(error: runner.CodeError) -> Outcome:
    """No marks, left for a person, as the marker's code failed."""
    return code_failure_outcome(as_sentence(f"Your code failed: {error}"))


def code_problems(
    code: str, field: str = "/code", mode: Literal["exec", "eval"] = "exec"
) -> Iterator[Problem]:
    """Code at ``field`` that can't run at all, at the line at fault."""
    try:
        compile(code, "<code>", mode)
    except SyntaxError as error:
        line_note = f"Line {error.lineno}: " if error.lineno else ""
        yield Problem(as_sentence(f"{line_note}{error.msg}"), field=field)


class CustomCode(AnswerMethod[CustomCodeSettings]):
    name = "gradeflow/custom-code"
    version = 1
    label = "Custom Code"
    summary = "Your Python code decides the marks."
    guide = (
        "For a marking rule no other method covers, written in Python.\n\n"
        "Your code runs once for each answer, with the answer in `answer` as the question reads "
        "it: text for a text or code question, a number for a number question, a set of options "
        "for a choice question and a list for a values question. The question's **Reading "
        "answers** apply first, so text is trimmed and lowercased by default. Blank answers get no "
        "marks without reaching your code.\n\n"
        "Set `correct` to `True` or `False`, or `proportion` to a number from 0 to 1 for part "
        "marks. Set `feedback` to tell the student why. They see it as plain text, and see nothing "
        "if you don't set it. If your code raises an error or runs past its **Time limit**, the "
        "answer needs marking, with the error as the reason.\n\n"
        "```python\n"
        'correct = answer in {"h2o", "water"}\n'
        "if not correct:\n"
        '    feedback = "Expected H2O."\n'
        "```\n\n"
        "Each of the **Parameters** is a variable with that name in your code, of the type you "
        "give it. Put values you may change there, such as a threshold, to change them without "
        "editing the code.\n\n"
        "A code answer of several files comes as one text, each file after a line such as `-- "
        "board.py --`.\n\n"
        "To mark several questions together in code, use Custom Code for a Group."
    )
    level = "advanced"
    question_types = ALL_QUESTION_TYPES
    settings = CustomCodeSettings
    examples = examples_beside(__file__)

    def settings_model_at(self, position: Position) -> type[Settings]:
        question = position.question_model
        answer_description = (
            f"`answer` ({question.python_type().removesuffix(' | None')})"
            if question
            else "`answer`"
        )
        return specialized(
            self.settings, code={"description": code_description(answer_description)}
        )

    def describe(self, settings: CustomCodeSettings) -> str:
        return f"Python code decides the marks:\n\n{code_block(settings.code)}"

    def problems(self, settings: CustomCodeSettings, position: Position) -> Iterator[Problem]:
        return code_problems(settings.code)

    def mark_answer(
        self, settings: CustomCodeSettings, answer: Answer, question: Question
    ) -> Outcome:
        try:
            returned_variables = runner.run(
                settings.code,
                {"answer": answer, **parameter_values(settings.parameters)},
                RESULT_VARIABLES,
                time_limit_s=settings.time_limit,
                memory_limit_mb=settings.memory_limit,
            )
        except runner.CodeError as error:
            return code_error_outcome(error)
        return outcome_from_variables(returned_variables)
