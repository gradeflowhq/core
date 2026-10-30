from collections.abc import Mapping
from typing import Any

import pytest

from gradeflow_core import BUILTINS, Question, QuestionSet
from gradeflow_core.editing import Position
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from .helpers import describe, mark, mark_all, mark_group, problems_of, scheme

METHODS = {method.name: method for method in BUILTINS}


def code(source: str, **parameters: Mapping[str, Any]) -> dict[str, object]:
    return {"uses": "gradeflow/custom-code@1", "with": {"code": source, "parameters": parameters}}


def group(source: str, *questions: str) -> dict[str, object]:
    return {
        "uses": "gradeflow/custom-code-for-a-group@1",
        "with": {"questions": list(questions), "code": source},
    }


def test_marks_come_from_correct_or_proportion_with_the_feedback_set() -> None:
    correct = mark(code("correct = answer == 'yes'\nfeedback = 'Read ' + answer"), "yes")
    assert (correct.proportion, correct.feedback, correct.method) == (1, "Read yes", "Custom Code")
    assert mark(code("proportion = 0.25"), "x").proportion == 0.25
    assert mark(code("x = 1"), "x").proportion == 0


def test_code_reads_the_answer_as_the_question_reads_it() -> None:
    assert mark(code("correct = answer == 9.81"), "9.81 m/s", NumberQuestion()).correct
    assert mark(
        code("correct = answer == {'a', 'b'}"),
        "b, a",
        ChoiceQuestion(options=frozenset({"a", "b"})),
    ).correct


def test_parameters_are_variables() -> None:
    words = {
        "type": "list",
        "value": [{"type": "text", "value": "light"}, {"type": "text", "value": "water"}],
    }
    assert mark(
        code(
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
    result = mark(code(source), "x")
    assert (result.proportion, result.marked) == (0, False)
    assert message in result.feedback


def test_code_that_cant_run_is_a_problem_of_the_check() -> None:
    assert problems_of(code("correct = (\n")) == [("", ("code",), "Line 1: '(' was never closed.")]
    [marking] = scheme(group("results = {}\nif True\n", "Q1")).markings
    [problem] = marking.problems(QuestionSet(QUESTIONS))
    assert (problem.check, problem.field, problem.message) == (
        "",
        ("code",),
        "Line 2: expected ':'.",
    )


def test_the_starting_code_gives_no_marks() -> None:
    [marking] = scheme({"question": "Q1", "uses": "gradeflow/custom-code@1"}).markings
    assert marking.with_.code == 'correct = False\nfeedback = ""\n'  # type: ignore[attr-defined]
    assert mark_all({"uses": "gradeflow/custom-code@1"}, ["x"])[0].proportion == 0
    assert describe(code("correct = True")) == (
        "Python code decides the marks:\n\n```python\ncorrect = True\n```"
    )


QUESTIONS: dict[str, Question] = {
    "Q1": TextQuestion(),
    "Q2": NumberQuestion(),
    "Q3": TextQuestion(),
}


def code_help(method: str, position: Position) -> str:
    help_text: str = METHODS[method].settings_schema(position)["properties"]["code"]["description"]
    return help_text


def test_the_code_help_names_the_type_each_answer_comes_as() -> None:
    questions = QuestionSet({"Q1": TextQuestion(), "Q2": NumberQuestion()})
    assert "Reads answer (int | float | None)" in code_help(
        "gradeflow/custom-code", Position(questions, "Q2")
    )
    assert "Reads answers (Q1: str | None, Q2: int | float | None)" in code_help(
        "gradeflow/custom-code-for-a-group", Position(questions, in_group=True)
    )


def test_group_code_marks_each_of_its_questions_from_its_results() -> None:
    source = (
        "results = {\n"
        "    'Q1': {'correct': answers['Q1'] == 'a', 'feedback': 'one'},\n"
        "    'Q2': {'proportion': 0.5},\n"
        "}\n"
        "seen = sorted(answers)\n"
        "assert seen == ['Q1', 'Q2'], seen\n"
    )
    results = mark_group(group(source, "Q1", "Q2"), {"Q1": "a", "Q2": "1", "Q3": "c"}, QUESTIONS)
    assert {key: (r.proportion, r.feedback) for key, r in results.items()} == {
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
    results = mark_group(group(source, "Q1", "Q2"), {"Q1": "a", "Q2": "1"}, QUESTIONS)
    assert not results["Q2"].marked
    assert message in results["Q2"].feedback


def test_group_code_lists_each_question_of_the_assessment_once() -> None:
    [marking] = scheme(group("results = {}", "Q1", "Q1", "Q9")).markings
    found = [(p.check, p.field, p.message) for p in marking.problems(QuestionSet(QUESTIONS))]
    assert found == [
        ("", ("questions", 1), "Q1 is listed twice."),
        ("", ("questions", 2), "Q9 isn't in the supplied questions."),
    ]
    assert problems_of(code("correct = True")) == []


def test_a_check_with_parameters_written_without_its_defaults_reads_back() -> None:
    check = code("correct = n == 2", n={"type": "integer", "value": 2})
    [marking] = scheme({"question": "Q1", **check}).markings
    assert marking.dump()["with"]["parameters"] == {"n": {"type": "integer", "value": 2}}
    assert mark(marking.dump(), "x").correct
