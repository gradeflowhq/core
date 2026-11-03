from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from gradeflow_core import BUILTINS, Question, QuestionSet
from gradeflow_core.editing import Position
from gradeflow_core.methods.code import runner
from gradeflow_core.methods.code.custom_code import CustomCodeSettings
from gradeflow_core.methods.code.custom_code_for_a_group import CustomCodeForAGroupSettings
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from .helpers import describe, mark, mark_all, mark_group, mark_scheme_of, problems_of

METHODS = {method.name: method for method in BUILTINS}


def custom_code(source: str, **parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {"uses": "gradeflow/custom-code@1", "with": {"code": source, "parameters": parameters}}


def custom_code_for_a_group(source: str, *questions: str) -> dict[str, Any]:
    return {
        "uses": "gradeflow/custom-code-for-a-group@1",
        "with": {"questions": list(questions), "code": source},
    }


def test_marks_come_from_correct_or_proportion_with_the_feedback_set() -> None:
    result = mark(custom_code("correct = answer == 'yes'\nfeedback = 'Read ' + answer"), "yes")
    assert (result.proportion, result.feedback, result.method_label) == (
        1,
        "Read yes",
        "Custom Code",
    )
    assert mark(custom_code("proportion = 0.25"), "x").proportion == 0.25
    assert mark(custom_code("x = 1"), "x").proportion == 0


def test_code_reads_the_answer_as_the_question_reads_it() -> None:
    assert mark(custom_code("correct = answer == 9.81"), "9.81 m/s", NumberQuestion()).correct
    assert mark(
        custom_code("correct = answer == {'a', 'b'}"),
        "b, a",
        ChoiceQuestion(options=frozenset({"a", "b"})),
    ).correct


def test_parameters_are_variables() -> None:
    words = {
        "type": "list",
        "value": [{"type": "text", "value": "light"}, {"type": "text", "value": "water"}],
    }
    assert mark(
        custom_code(
            "correct = answer in words and limit == 2",
            words=words,
            limit={"type": "integer", "value": 2},
        ),
        "water",
    ).correct


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("raise ValueError('bad')", "Your code failed: ValueError: bad."),
        ("proportion = 2", "Your code set proportion to 2. It must be a number from 0 to 1."),
        (
            "proportion = 'all'",
            "Your code set proportion to “all”. It must be a number from 0 to 1.",
        ),
    ],
)
def test_code_that_fails_leaves_the_answer_for_a_person(source: str, message: str) -> None:
    result = mark(custom_code(source), "x")
    assert (result.proportion, result.is_marked) == (0, False)
    assert message in result.feedback


def test_code_that_cant_run_is_a_problem_of_the_check() -> None:
    assert problems_of(custom_code("correct = (\n")) == [
        ("", ("code",), "Line 1: '(' was never closed.")
    ]
    [marking] = mark_scheme_of(custom_code_for_a_group("results = {}\nif True\n", "Q1")).markings
    [problem] = marking.problems(QuestionSet(QUESTIONS))
    assert (problem.check, problem.field, problem.message) == (
        "",
        ("code",),
        "Line 2: expected ':'.",
    )


def test_the_starting_code_gives_no_marks() -> None:
    [marking] = mark_scheme_of({"question": "Q1", "uses": "gradeflow/custom-code@1"}).markings
    assert marking.with_.code == 'correct = False\nfeedback = ""\n'  # type: ignore[attr-defined]
    assert mark_all({"uses": "gradeflow/custom-code@1"}, ["x"])[0].proportion == 0
    assert describe(custom_code("correct = True")) == (
        "Python code decides the marks:\n\n```python\ncorrect = True\n```"
    )


QUESTIONS: dict[str, Question] = {
    "Q1": TextQuestion(),
    "Q2": NumberQuestion(),
    "Q3": TextQuestion(),
}


def code_description(method: str, position: Position) -> str:
    help_text: str = METHODS[method].settings_schema(position)["properties"]["code"]["description"]
    return help_text


def test_the_code_description_names_the_type_each_answer_comes_as() -> None:
    questions = QuestionSet({"Q1": TextQuestion(), "Q2": NumberQuestion()})
    assert "Reads answer (int | float | None)" in code_description(
        "gradeflow/custom-code", Position(questions, "Q2")
    )
    assert "Reads answers (Q1: str | None, Q2: int | float | None)" in code_description(
        "gradeflow/custom-code-for-a-group", Position(questions, in_group=True)
    )


def with_limits(check: dict[str, Any], **limits: int) -> dict[str, Any]:
    return {**check, "with": {**check["with"], **limits}}


def test_code_runs_within_its_memory_limit() -> None:
    source = "data = b'x' * (96 * 1024**2)\ncorrect = True"
    assert mark(with_limits(custom_code(source), memory_limit=256), "x").proportion == 1
    result = mark(with_limits(custom_code(source), memory_limit=64), "x")
    assert (result.is_marked, result.feedback) == (False, "Your code failed: MemoryError.")
    group_check = with_limits(custom_code_for_a_group(source, "Q1"), memory_limit=64)
    assert "MemoryError" in mark_group(group_check, {"Q1": "a"}, QUESTIONS)["Q1"].feedback


def test_code_runs_within_its_time_limit() -> None:
    result = mark(with_limits(custom_code("while True: pass"), time_limit=1), "x")
    assert (result.is_marked, result.feedback) == (
        False,
        "Your code failed: it took longer than 1 second.",
    )
    group_check = with_limits(custom_code_for_a_group("while True: pass", "Q1"), time_limit=1)
    assert "longer than 1 second" in mark_group(group_check, {"Q1": "a"}, QUESTIONS)["Q1"].feedback


@pytest.mark.parametrize(
    "limits",
    [
        {"time_limit": 0},
        {"time_limit": runner.MAX_TIME_LIMIT_S + 1},
        {"memory_limit": runner.MIN_MEMORY_LIMIT_MB - 1},
        {"memory_limit": runner.MAX_MEMORY_LIMIT_MB + 1},
    ],
)
def test_limits_are_bounded(limits: dict[str, int]) -> None:
    for settings, valid_settings in (
        (CustomCodeSettings, {"code": "x = 1"}),
        (CustomCodeForAGroupSettings, {"code": "x = 1", "questions": ["Q1"]}),
    ):
        settings.model_validate(valid_settings)
        with pytest.raises(ValidationError) as caught:
            settings.model_validate({**valid_settings, **limits})
        assert [error_detail["loc"] for error_detail in caught.value.errors()] == [tuple(limits)]


def test_group_code_marks_each_of_its_questions_from_its_results() -> None:
    source = (
        "results = {\n"
        "    'Q1': {'correct': answers['Q1'] == 'a', 'feedback': 'one'},\n"
        "    'Q2': {'proportion': 0.5},\n"
        "}\n"
        "question_ids = sorted(answers)\n"
        "assert question_ids == ['Q1', 'Q2'], question_ids\n"
    )
    results = mark_group(
        custom_code_for_a_group(source, "Q1", "Q2"), {"Q1": "a", "Q2": "1", "Q3": "c"}, QUESTIONS
    )
    assert {key: (result.proportion, result.feedback) for key, result in results.items()} == {
        "Q1": (1, "one"),
        "Q2": (0.5, ""),
    }


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("raise ValueError('bad')", "ValueError: bad"),
        ("results = 1", "Your code must set results to a dict by question."),
        ("results = {'Q1': {'correct': True}}", "Your code didn't set results for Q2."),
        (
            "results = {'Q1': {}, 'Q2': {'proportion': 3}}",
            "Your code set proportion for Q2 to 3. It must be a number from 0 to 1.",
        ),
    ],
)
def test_group_code_that_fails_leaves_its_questions_for_a_person(source: str, message: str) -> None:
    results = mark_group(
        custom_code_for_a_group(source, "Q1", "Q2"), {"Q1": "a", "Q2": "1"}, QUESTIONS
    )
    assert not results["Q2"].is_marked
    assert message in results["Q2"].feedback


def test_group_code_lists_each_question_of_the_assessment_once() -> None:
    [marking] = mark_scheme_of(custom_code_for_a_group("results = {}", "Q1", "Q1", "Q9")).markings
    problems = [
        (problem.check, problem.field, problem.message)
        for problem in marking.problems(QuestionSet(QUESTIONS))
    ]
    assert problems == [
        ("", ("questions", 1), "Q1 is listed twice."),
        ("", ("questions", 2), "Q9 isn't one of the questions."),
    ]
    assert problems_of(custom_code("correct = True")) == []


def test_a_check_with_parameters_written_without_its_defaults_reads_back() -> None:
    check = custom_code("correct = n == 2", n={"type": "integer", "value": 2})
    [marking] = mark_scheme_of({"question": "Q1", **check}).markings
    assert marking.dump()["with"]["parameters"] == {"n": {"type": "integer", "value": 2}}
    assert mark(marking.dump(), "x").correct
