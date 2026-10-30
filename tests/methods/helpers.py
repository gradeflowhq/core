from collections.abc import Mapping
from typing import Any

import gradeflow_core
from gradeflow_core import (
    BUILTINS,
    MarkScheme,
    Methods,
    Question,
    QuestionResult,
    QuestionSet,
    Submission,
)
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.text import TextQuestion

QID = "Q1"
CODE = CodeQuestion()


def scheme(*markings: Mapping[str, Any], methods: Methods = BUILTINS) -> MarkScheme:
    return MarkScheme.load(list(markings), methods=methods)


def accepted(answer: str, **keys: Any) -> dict[str, Any]:
    return {**keys, "uses": "gradeflow/accepted-answers@1", "with": {"answers": [answer]}}


def key_point(point: str, **keys: Any) -> dict[str, Any]:
    return {**keys, "uses": "gradeflow/key-point@1", "with": {"point": point}}


def code_tests(*tests: Mapping[str, str], **settings: Any) -> dict[str, Any]:
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
    questions = QuestionSet({QID: question or TextQuestion()})
    marking = scheme({"question": QID, **check}, methods=methods)
    submissions = [
        Submission(student_id=str(index), answers={QID: answer})
        for index, answer in enumerate(answers)
    ]
    marked = gradeflow_core.mark(questions, marking, submissions, **options)
    return [submission.results[QID] for submission in marked]


def mark(check: Mapping[str, Any], answer: str, question: Question | None = None) -> QuestionResult:
    [result] = mark_all(check, [answer], question)
    return result


def mark_group(
    marking: Mapping[str, Any], answers: Mapping[str, str], questions: Mapping[str, Question]
) -> dict[str, QuestionResult]:
    question_set = QuestionSet(dict(questions))
    [submission] = gradeflow_core.mark(
        question_set, scheme(marking), [Submission(student_id="s", answers=dict(answers))]
    )
    return submission.results


def problems_of(
    check: Mapping[str, Any], question: Question | None = None
) -> list[tuple[str, tuple[Any, ...], str]]:
    questions = QuestionSet({QID: question or TextQuestion()})
    [marking] = scheme({"question": QID, **check}).markings
    return [
        (problem.check, problem.field, problem.message) for problem in marking.problems(questions)
    ]


def proportions(results: list[QuestionResult]) -> list[float]:
    return [round(result.proportion, 4) for result in results]


def describe(check: Mapping[str, Any]) -> str:
    [marking] = scheme({"question": QID, **check}).markings
    return marking.describe()
