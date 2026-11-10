from collections.abc import Iterable, Mapping
from typing import Any

import gradeflow_core
from gradeflow_core import (
    BUILTINS,
    MarkScheme,
    Methods,
    Problem,
    Question,
    QuestionResult,
    QuestionSet,
    Submission,
)
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.text import TextQuestion

QUESTION_ID = "Q1"
CODE_QUESTION = CodeQuestion()
WRAPPED_NUMBER_RANGE = (
    "sample/wrapped",
    """label: Wrapped
version: 1
summary: A wrapped number range.
question_types: [text]
steps:
  - uses: gradeflow/number-range@1
    with: {min_value: 1}
""",
)
"""A method file whose check can't mark the text questions it says it marks."""


def mark_scheme_of(*markings: Mapping[str, Any], methods: Methods = BUILTINS) -> MarkScheme:
    return MarkScheme.load(list(markings), methods=methods)


def accepted_answer(answer: str, **check_keys: Any) -> dict[str, Any]:
    return {**check_keys, "uses": "gradeflow/accepted-answers@1", "with": {"answers": [answer]}}


def key_point(point: str, **check_keys: Any) -> dict[str, Any]:
    return {**check_keys, "uses": "gradeflow/key-point@1", "with": {"point": point}}


def workflow(marks_by: str, *steps: Mapping[str, Any], **step_keys: Any) -> dict[str, Any]:
    return {
        **step_keys,
        "uses": "gradeflow/workflow@1",
        "with": {"marks_by": marks_by, "steps": list(steps)},
    }


def with_value_inputs(*checks: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``checks`` as Multiple Values holds them, each reading the value in its place."""
    return [{**check, "input": f"{{{{ values[{index}] }}}}"} for index, check in enumerate(checks)]


def code_tests(*tests: Mapping[str, Any], **settings: Any) -> dict[str, Any]:
    return {"uses": "gradeflow/code-tests@1", "with": {"tests": list(tests), **settings}}


def mark_all(
    check: Mapping[str, Any],
    answers: list[str],
    question: Question | None = None,
    methods: Methods = BUILTINS,
    **options: Any,
) -> list[QuestionResult]:
    """How ``check`` marks each answer to Q1, a text question unless ``question`` says, with the
    ``options`` of ``mark``."""
    questions = QuestionSet({QUESTION_ID: question or TextQuestion()})
    mark_scheme = mark_scheme_of({"question": QUESTION_ID, **check}, methods=methods)
    submissions = [
        Submission(student_id=str(index), answers={QUESTION_ID: answer})
        for index, answer in enumerate(answers)
    ]
    marked_submissions = gradeflow_core.mark(questions, mark_scheme, submissions, **options)
    return [submission.results[QUESTION_ID] for submission in marked_submissions]


def mark(check: Mapping[str, Any], answer: str, question: Question | None = None) -> QuestionResult:
    [result] = mark_all(check, [answer], question)
    return result


def mark_group(
    marking: Mapping[str, Any], answers: Mapping[str, str], questions: Mapping[str, Question]
) -> dict[str, QuestionResult]:
    question_set = QuestionSet(dict(questions))
    [submission] = gradeflow_core.mark(
        question_set, mark_scheme_of(marking), [Submission(student_id="s", answers=dict(answers))]
    )
    return submission.results


def located(problems: Iterable[Problem]) -> list[tuple[str, tuple[str | int, ...], str]]:
    """Each problem as the check and field it is at, and its message."""
    return [(problem.step_path, problem.field, problem.message) for problem in problems]


def problems_of(
    check: Mapping[str, Any], question: Question | None = None
) -> list[tuple[str, tuple[str | int, ...], str]]:
    questions = QuestionSet({QUESTION_ID: question or TextQuestion()})
    [marking] = mark_scheme_of({"question": QUESTION_ID, **check}).markings
    return located(marking.problems(questions))


def proportions(results: list[QuestionResult]) -> list[float]:
    return [round(result.proportion, 4) for result in results]


def describe(check: Mapping[str, Any]) -> str:
    [marking] = mark_scheme_of({"question": QUESTION_ID, **check}).markings
    return marking.describe()
