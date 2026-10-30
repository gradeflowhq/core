from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import Field

from ...common.hints import CODE, hint
from ...common.problems import Problem
from ...common.text import fenced, markdown_join
from ...questions.types import QuestionId, Unreadable
from ..check import Check, Context
from ..example import examples_beside
from ..method import Method
from ..outcome import Outcome
from ..position import Position
from ..settings import Settings, literal, specialised
from .custom_code import MARKS, code_problems, failed, outcome_of, run
from .parameters import PARAMETERS, Parameter, ParameterName, values

STARTER = (
    "results = {}\n"
    "for question, answer in answers.items():\n"
    '    results[question] = {"correct": False, "feedback": ""}\n'
)


def code_help(answers: str) -> str:
    return f"Reads {answers} and any parameters, and sets results[question] to a dict of {MARKS}."


class CustomCodeForAGroupSettings(Settings):
    questions: list[QuestionId] = Field(
        min_length=1,
        title="Questions",
        description="The questions this code marks.",
        json_schema_extra=hint(kind="question", item="question"),
    )
    code: str = Field(
        default=STARTER,
        min_length=1,
        title="Code",
        description=code_help("answers, by question,"),
        json_schema_extra=hint(**CODE),
    )
    parameters: dict[ParameterName, Parameter] = PARAMETERS


class CustomCodeForAGroup(Method[CustomCodeForAGroupSettings]):
    name = "gradeflow/custom-code-for-a-group"
    version = 1
    label = "Custom Code for a Group"
    summary = "Your Python code marks several questions together."
    guide = (
        "**Use it when:** The marks for several questions depend on each other in a way no other "
        "method covers, and you can write Python.\n\n"
        "**How it marks:** Your code gets every answer in `answers`, by question, and sets "
        "`results` to give each question `correct` or `proportion`, and `feedback` if you like.\n\n"
        "**Good to know:** Put values you may change in **Parameters**, so you can change them "
        "without editing the code.\n\n"
        "**Related:** For one question, use Custom Code."
    )
    level = "advanced"
    group = True
    settings = CustomCodeForAGroupSettings
    examples = examples_beside(__file__)

    def settings_at(self, position: Position) -> type[Settings]:
        keys = tuple(position.questions)
        annotation = list[literal(keys)] if keys else list[str]  # type: ignore[misc]
        kinds = ", ".join(
            f"{key}: {question.python_type()}" for key, question in position.questions.items()
        )
        answers = f"answers ({kinds})" if kinds else "answers, by question,"
        return specialised(
            self.settings,
            questions={"annotation": annotation},
            code={"description": code_help(answers)},
        )

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        listed = markdown_join(settings.questions, conjunction="and")
        return f"Python code decides the marks for {listed}:\n\n{fenced(settings.code)}"

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        settings = self._settings_of(check)
        given = {key: context.answers.get(key) for key in settings.questions}
        variables: dict[str, Any] = {
            "answers": {key: None if isinstance(a, Unreadable) else a for key, a in given.items()},
            **values(settings.parameters),
        }
        if (error := run(settings.code, variables)) is not None:
            return dict.fromkeys(settings.questions, failed(error))
        results = variables.get("results")
        if not isinstance(results, Mapping):
            return dict.fromkeys(
                settings.questions, failed("Your code must set results to a dict by question.")
            )
        return {key: _outcome(results.get(key), key) for key in settings.questions}

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return set(self._settings_of(check).questions)

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        yield from code_problems(settings.code)
        listed = settings.questions
        for index, key in enumerate(listed):
            if key in listed[:index]:
                yield Problem(f"{key} is listed twice.", field=("questions", index))
            elif key not in position.questions:
                yield Problem(f"{key} isn't in the supplied questions.", field=("questions", index))


def _outcome(result: object, key: QuestionId) -> Outcome:
    if result is None:
        return failed(f"Your code didn't set results for {key}.")
    if not isinstance(result, Mapping):
        return failed(
            f"Your code must set results[{key!r}] to a dict of correct or proportion, and feedback."
        )
    return outcome_of(result, key)
