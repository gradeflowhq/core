from collections.abc import Iterator, Mapping
from typing import Any

from pydantic import Field

from ...common.hints import PYTHON_CODE_HINT, hint
from ...common.problems import Problem
from ...common.text import code_block, code_join
from ...questions.types import QuestionId, Unreadable
from ..check import Check, Context
from ..example import examples_beside
from ..method import Method
from ..outcome import Outcome
from ..position import Position
from ..settings import Settings, literal, specialized
from .custom_code import (
    MEMORY_LIMIT_FIELD,
    RESULT_VARIABLES_HELP,
    TIME_LIMIT_FIELD,
    code_failure,
    code_problems,
    outcome_from_variables,
    run_failure,
)
from .parameters import PARAMETERS_FIELD, Parameter, ParameterName, parameter_values

STARTER_CODE = (
    "results = {}\n"
    "for question, answer in answers.items():\n"
    '    results[question] = {"correct": False, "feedback": ""}\n'
)


def group_code_description(answers: str) -> str:
    return (
        f"Reads {answers} and any parameters, and sets results[question] to a dict of "
        f"{RESULT_VARIABLES_HELP}."
    )


class CustomCodeForAGroupSettings(Settings):
    questions: list[QuestionId] = Field(
        min_length=1,
        title="Questions",
        description="The questions this code marks.",
        json_schema_extra=hint(kind="question", item_label="question"),
    )
    code: str = Field(
        default=STARTER_CODE,
        min_length=1,
        title="Code",
        description=group_code_description("answers, by question,"),
        json_schema_extra=hint(**PYTHON_CODE_HINT),
    )
    parameters: dict[ParameterName, Parameter] = PARAMETERS_FIELD
    time_limit: int = TIME_LIMIT_FIELD
    memory_limit: int = MEMORY_LIMIT_FIELD


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
    marks_question_group = True
    settings = CustomCodeForAGroupSettings
    examples = examples_beside(__file__)

    def settings_model_at(self, position: Position) -> type[Settings]:
        question_ids = tuple(position.questions)
        annotation = list[literal(question_ids)] if question_ids else list[str]  # type: ignore[misc]
        typed_questions = ", ".join(
            f"{question_id}: {question.python_type()}"
            for question_id, question in position.questions.items()
        )
        answers = f"answers ({typed_questions})" if typed_questions else "answers, by question,"
        return specialized(
            self.settings,
            questions={"annotation": annotation},
            code={"description": group_code_description(answers)},
        )

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        question_list = code_join(settings.questions, conjunction="and")
        return f"Python code decides the marks for {question_list}:\n\n{code_block(settings.code)}"

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        settings = self._settings_of(check)
        answers = {
            question_id: context.answers.get(question_id) for question_id in settings.questions
        }
        variables: dict[str, Any] = {
            "answers": {
                question_id: None if isinstance(answer, Unreadable) else answer
                for question_id, answer in answers.items()
            },
            **parameter_values(settings.parameters),
        }
        if (
            error := run_failure(
                settings.code, variables, settings.time_limit, settings.memory_limit
            )
        ) is not None:
            return dict.fromkeys(settings.questions, code_failure(error))
        results = variables.get("results")
        if not isinstance(results, Mapping):
            return dict.fromkeys(
                settings.questions,
                code_failure("Your code must set results to a dict by question."),
            )
        return {
            question_id: _question_outcome(results.get(question_id), question_id)
            for question_id in settings.questions
        }

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return set(self._settings_of(check).questions)

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        yield from code_problems(settings.code)
        questions = settings.questions
        for index, question_id in enumerate(questions):
            if question_id in questions[:index]:
                yield Problem(f"{question_id} is listed twice.", field=("questions", index))
            elif question_id not in position.questions:
                yield Problem(
                    f"{question_id} isn't one of the questions.", field=("questions", index)
                )


def _question_outcome(result: object, question_id: QuestionId) -> Outcome:
    if result is None:
        return code_failure(f"Your code didn't set results for {question_id}.")
    if not isinstance(result, Mapping):
        return code_failure(
            f"Your code must set results[{question_id!r}] to a dict of correct or proportion, "
            "and feedback."
        )
    return outcome_from_variables(result, question_id)
