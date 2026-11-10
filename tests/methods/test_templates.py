from collections.abc import Mapping
from unittest.mock import Mock, call

import pytest

from gradeflow_core.methods import templates
from gradeflow_core.methods.outcome import Outcome
from gradeflow_core.methods.templates import (
    CHECK_RESULT_SHAPE,
    PART_RESULT_SHAPE,
    ListOf,
    Shape,
    TemplateError,
    condition,
    evaluate,
    reference,
    render,
    validate,
)
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from .helpers import accepted_answer, mark, mark_all, mark_group, problems_of

TEMPLATE_SCOPE: Mapping[str, Shape] = {
    "steps": {"test": CHECK_RESULT_SHAPE},
    "settings": {"points": ListOf("text")},
    "parts": ListOf(PART_RESULT_SHAPE),
}


def test_a_reference_is_a_value_that_is_exactly_one_pair_of_braces() -> None:
    assert reference(" {{ settings.points }} ") == "settings.points"
    assert reference("Has {{ settings.points }}") is None
    assert evaluate("settings.n", {"settings": {"n": [1, 2]}}) == [1, 2]


def test_a_condition_is_one_reference() -> None:
    assert condition("{{ not steps.test.output.correct }}") == "not steps.test.output.correct"
    with pytest.raises(TemplateError, match="Write a condition as one"):
        condition("{{ steps.test.output.correct }} and y")


def test_templates_branch_loop_and_use_the_filters() -> None:
    source = (
        "{% if proportion > 0.5 or not items %}Good{% elif proportion > 0 and items %}Some"
        "{% else %}None{% endif %}: "
        "{{ items | and_list }}; {{ items | quoted | or_list }}; {{ proportion | percent }}; "
        "{% for item in items %}{{ loop.index }}{% endfor %}"
    )
    assert (
        render(source, {"proportion": 0.25, "items": ["a", "b", "c"]})
        == "Some: a, b and c; “a”, “b” or “c”; 25%; 123"
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("{% set x = 1 %}", "A template can't use {% set %}."),
        ("{% macro m() %}{% endmacro %}", "A template can't use {% macro %}."),
        ('{% include "x" %}', "A template can't use {% include %}."),
        ('{% import "x" as y %}', "A template can't use {% import %}."),
        (
            "{{ settings.points.pop() }}",
            "A template can't call anything; use a filter, such as {{ items | and_list }}.",
        ),
        (
            "{{ parts | length * 2 }}",
            "A template can't do arithmetic; use a filter, such as {{ proportion | percent }}.",
        ),
        ("{{ parts | wordcount }}", "There's no filter wordcount."),
        ("{{ answer }}", "There's no answer."),
        ("{{ steps.test.output.pass }}", "steps.test.output has no pass."),
        ("{{ steps.test.output.correct.x }}", "steps.test.output.correct is bool, which has no x."),
        ("{% for part in parts %}{{ part.nam }}{% endfor %}", "part has no nam."),
        ('{{ "a".__class__ }}', "There's no __class__."),
        ("{% if %}", "Expected an expression, got 'end of statement block' (line 1)"),
    ],
)
def test_templates_are_refused_on_load_for_what_they_cant_do(source: str, message: str) -> None:
    with pytest.raises(TemplateError) as exc_info:
        validate(source, TEMPLATE_SCOPE)
    assert str(exc_info.value) == message


def test_a_checks_feedback_template_replaces_its_feedback_with_the_answer_as_written() -> None:
    check = accepted_answer(
        "Paris",
        feedback=(
            "{% if correct %}Yes, {{ answer }}.{% else %}{{ answer }} isn't the capital.{% endif %}"
        ),
    )
    assert [mark(check, answer).feedback for answer in ("Paris ", "Lyon")] == [
        "Yes, Paris.",
        "Lyon isn't the capital.",
    ]


def test_feedback_for_each_outcome_replaces_its_feedback_and_an_empty_one_keeps_it() -> None:
    check = accepted_answer("Paris", feedback={"correct": "Yes, {{ answer }}.", "incorrect": ""})
    assert [mark(check, answer).feedback for answer in ("paris", "lyon")] == [
        "Yes, paris.",
        "Expected “Paris”.",
    ]


def test_feedback_for_an_outcome_says_where_its_template_is_wrong() -> None:
    check = accepted_answer("Paris", feedback={"partly_correct": "{{ anser }}"})
    assert problems_of(check) == [("", ("feedback", "partly_correct"), "There's no anser.")]


def test_feedback_can_name_earlier_checks_results() -> None:
    marking = {
        "uses": "gradeflow/workflow@1",
        "with": {
            "steps": [
                {
                    "id": "unit",
                    "question": "Q1",
                    "weight": 0,
                    "uses": "gradeflow/key-point@1",
                    "with": {"point": "m/s"},
                },
                accepted_answer(
                    "9.81",
                    question="Q2",
                    feedback=(
                        "{% if not steps.unit.output.correct %}Give units in Q1. {% endif %}"
                        "{{ proportion | percent }}"
                    ),
                ),
            ]
        },
    }
    [result] = mark_group(
        marking, {"Q1": "9.81", "Q2": "9.81"}, {"Q1": TextQuestion(), "Q2": NumberQuestion()}
    ).values()
    assert result.feedback == "Give units in Q1. 100%"


def test_a_marking_with_a_template_it_cant_use_says_where() -> None:
    check = {"uses": "gradeflow/key-point@1", "with": {"point": "a"}, "feedback": "{{ score }}"}
    assert problems_of(check) == [("", ("feedback",), "There's no score.")]
    workflow = {
        "uses": "gradeflow/workflow@1",
        "with": {"steps": [{**check, "feedback": None, "if": "{{ steps.later.output.correct }}"}]},
    }
    assert problems_of(workflow) == [
        ("with.steps.0", ("if",), "No step before this one is called later.")
    ]


def test_a_condition_that_doesnt_parse_says_so_plainly() -> None:
    check = {"uses": "gradeflow/key-point@1", "with": {"point": "a"}, "if": "{{ x garbage(( }}"}
    assert problems_of(check) == [("", ("if",), "Can't read this condition.")]


@pytest.mark.parametrize("condition_text", ["{{ false }}", "{{ 1 > 0 }}"])
def test_a_condition_reads_an_earlier_step_or_a_setting(condition_text: str) -> None:
    check = {"uses": "gradeflow/key-point@1", "with": {"point": "a"}}
    message = (
        "Make a condition read an earlier step or a setting, such as "
        "{{ steps.test.output.correct }}."
    )
    assert problems_of({**check, "if": condition_text}) == [("", ("if",), message)]
    workflow = {
        "uses": "gradeflow/workflow@1",
        "with": {"steps": [check, {**check, "if": condition_text}]},
    }
    assert problems_of(workflow) == [("with.steps.1", ("if",), message)]


@pytest.mark.parametrize(
    ("field", "problem"),
    [
        ("correct", None),
        ("proportion", None),
        ("feedback", None),
        ("name", "steps.first.output has no name."),
        ("question", "steps.first.output has no question."),
    ],
)
def test_a_condition_reads_an_earlier_checks_correct_proportion_and_feedback(
    field: str, problem: str | None
) -> None:
    first = {
        "id": "first",
        "weight": 0,
        "uses": "gradeflow/key-point@1",
        "with": {"point": "a"},
    }
    second = {
        "if": f"{{{{ steps.first.output.{field} }}}}",
        "uses": "gradeflow/key-point@1",
        "with": {"point": "b"},
    }
    marking = {"uses": "gradeflow/workflow@1", "with": {"steps": [first, second]}}
    expected = [("with.steps.1", ("if",), problem)] if problem else []
    assert problems_of(marking) == expected


def test_feedback_can_add_to_the_methods_own() -> None:
    check = {
        "uses": "gradeflow/key-point@1",
        "with": {"point": "a"},
        "feedback": "{{ feedback }} See page 3.",
    }
    assert mark(check, "a").feedback == "Has “a”. See page 3."


def test_marking_compiles_each_template_once_for_every_student(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from_string = Mock(wraps=templates.ENVIRONMENT.from_string)
    monkeypatch.setattr(templates.ENVIRONMENT, "from_string", from_string)
    feedback = "{{ answer }} was marked {{ feedback }} once per template"
    check = {"uses": "gradeflow/key-point@1", "with": {"point": "x"}, "feedback": feedback}
    mark_all(check, [f"answer {index}" for index in range(5)])
    assert from_string.call_args_list.count(call(feedback)) == 1


def test_template_shapes_match_what_marking_gives_them() -> None:
    assert PART_RESULT_SHAPE.keys() == Outcome(1.0).template_values("Q1").keys()
    feedback = " ".join(f"{{{{ {name} }}}}" for name in templates.FEEDBACK_SCOPE)
    check = {"uses": "gradeflow/key-point@1", "with": {"point": "x"}, "feedback": feedback}
    [result] = mark_all(check, ["Paris"])
    assert result.feedback.startswith("Paris ")
