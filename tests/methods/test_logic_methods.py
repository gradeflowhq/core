from typing import Any

import pytest

from gradeflow_core import InvalidError
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

from .helpers import (
    accepted_answer,
    describe,
    mark_all,
    mark_group,
    mark_scheme_of,
    problems_of,
    proportions,
    with_value_inputs,
)


def test_not_allowed_gives_the_marks_its_check_doesnt() -> None:
    check = {
        "uses": "gradeflow/not-allowed@1",
        "with": {"check": {"uses": "gradeflow/key-point@1", "with": {"point": "denser"}}},
    }
    allowed, not_allowed = mark_all(check, ["less dense", "denser"])
    assert (allowed.proportion, not_allowed.proportion) == (1, 0)
    assert (allowed.feedback, not_allowed.feedback) == (
        "Correct.",
        "Not allowed. Includes “denser”.",
    )
    assert describe(check) == "**Not allowed:** Includes `denser`."


def test_conditional_describes_the_checks_of_each_branch_by_question() -> None:
    marking = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [accepted_answer("a", question="Q1")],
            "then": [accepted_answer("Paris", question="Q2")],
            "otherwise": [accepted_answer("Rome", question="Q2")],
        },
    }
    [conditional] = mark_scheme_of(marking).markings
    assert conditional.describe() == (
        "If all are correct:\n- **Q1:** Matches `a`.\n\n"
        "Then:\n- **Q2:** Matches `Paris`.\n\n"
        "Otherwise:\n- **Q2:** Matches `Rome`."
    )


def test_multiple_values_checks_each_value_in_order_or_in_any_order() -> None:
    checks = [
        {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["2"]}},
        {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["-3"]}},
    ]
    roots = ValuesQuestion(value_types=["number", "number"])
    in_order = {
        "uses": "gradeflow/multiple-values@1",
        "with": {"marks_by": "proportional", "checks": with_value_inputs(*checks)},
    }
    any_order = {
        "uses": "gradeflow/multiple-values@1",
        "with": {
            "marks_by": "proportional",
            "any_order": True,
            "checks": with_value_inputs(*checks),
        },
    }
    assert proportions(mark_all(in_order, ["2, -3", "-3, 2", "2, 5"], roots)) == [1, 0, 0.5]
    assert proportions(mark_all(any_order, ["2, -3", "-3, 2", "2, 5"], roots)) == [1, 1, 0.5]
    assert [result.feedback for result in mark_all(in_order, ["2, -3", "2, 5"], roots)] == [
        "Correct.",
        "Value 1: Correct.\nValue 2: Expected -3.",
    ]
    assert describe(in_order) == (
        "Each one counts:\n- **Value 1:** Matches `2`.\n- **Value 2:** Matches `-3`."
    )
    assert problems_of(
        {
            "uses": "gradeflow/multiple-values@1",
            "with": {"checks": checks},
        },
        roots,
    ) == [
        ("/with/checks/0", "", "Accepted Answers can't be used on a multiple values question."),
        ("/with/checks/1", "", "Accepted Answers can't be used on a multiple values question."),
    ]


@pytest.mark.parametrize("any_order", [False, True])
def test_multiple_values_needs_a_check_for_each_value(any_order: bool) -> None:
    roots = ValuesQuestion(value_types=["number", "number"])
    one_check = [accepted_answer("2")]
    three_checks = [accepted_answer("2"), accepted_answer("-3"), accepted_answer("5")]

    def problems_with(checks: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
        check = {
            "uses": "gradeflow/multiple-values@1",
            "with": {"any_order": any_order, "checks": with_value_inputs(*checks)},
        }
        return problems_of(check, roots)

    count_problem = ("", "/with/checks", "List 2 checks, one for each value.")
    assert problems_with(one_check) == [count_problem]
    assert problems_with(three_checks) == [
        count_problem,
        ("/with/checks/2", "/input", "The answer has 2 values."),
    ]


def test_multiple_values_in_any_order_need_values_of_one_type() -> None:
    check = {
        "uses": "gradeflow/multiple-values@1",
        "with": {
            "any_order": True,
            "checks": with_value_inputs(accepted_answer("Everest"), accepted_answer("8849")),
        },
    }
    assert problems_of(check, ValuesQuestion(value_types=["text", "number"])) == [
        ("", "/with/any_order", "Any order needs values of one type.")
    ]
    assert problems_of(check, ValuesQuestion(value_types=["text", "text"])) == []


def test_checks_in_a_check_of_one_value_read_it_unless_they_name_another() -> None:
    accepted_answers_method = {"uses": "gradeflow/accepted-answers@1"}
    heights = {
        "uses": "gradeflow/workflow@1",
        "with": {
            "steps": [
                {
                    "id": "k2",
                    "input": "{{ values[0] }}",
                    "weight": 0,
                    **accepted_answers_method,
                    "with": {"answers": ["K2"]},
                },
                {
                    "if": "{{ steps.k2.output.correct }}",
                    **accepted_answers_method,
                    "with": {"answers": ["8611"]},
                },
                {
                    "if": "{{ not steps.k2.output.correct }}",
                    **accepted_answers_method,
                    "with": {"answers": ["8849"]},
                },
            ]
        },
    }
    mountain = {"uses": "gradeflow/key-points@1", "with": {"points": ["Everest"]}}
    check = {
        "uses": "gradeflow/multiple-values@1",
        "with": {"marks_by": "proportional", "checks": with_value_inputs(mountain, heights)},
    }
    question = ValuesQuestion(value_types=["text", "text"])
    answers = ["Everest, 8849", "K2, 8611", "K2, Everest"]
    assert proportions(mark_all(check, answers, question)) == [1, 0.5, 0]
    assert problems_of(check, question) == []


def test_conditional_marks_its_branch_and_only_reads_its_condition() -> None:
    marking = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [
                {
                    "question": "Q1",
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": ["a"]},
                }
            ],
            "then": [
                {
                    "question": "Q2",
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": ["Paris"]},
                }
            ],
            "otherwise": [
                {
                    "question": "Q2",
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": ["Rome"]},
                }
            ],
        },
    }
    questions = {"Q1": TextQuestion(), "Q2": TextQuestion()}
    results = [
        mark_group(marking, answers, questions)
        for answers in (
            {"Q1": "a", "Q2": "Paris"},
            {"Q1": "b", "Q2": "Rome"},
            {"Q1": "a", "Q2": "Rome"},
        )
    ]
    assert [list(result) for result in results] == [["Q2"]] * 3
    assert [result["Q2"].proportion for result in results] == [1, 1, 0]
    assert results[0]["Q2"].method_label == "Conditional"


def test_answer_sets_count_the_set_giving_the_most_marks() -> None:
    marking = {
        "uses": "gradeflow/answer-sets@1",
        "with": {
            "sets": [
                {
                    "name": "Expected",
                    "uses": "gradeflow/workflow@1",
                    "with": {
                        "steps": [
                            accepted_answer("6", question="Q1"),
                            accepted_answer("24", question="Q2"),
                        ]
                    },
                },
                {
                    "name": "Read as 9",
                    "weight": 0.5,
                    "uses": "gradeflow/workflow@1",
                    "with": {
                        "steps": [
                            accepted_answer("9", question="Q1"),
                            accepted_answer("36", question="Q2"),
                        ]
                    },
                },
            ]
        },
    }
    questions = {"Q1": NumberQuestion(), "Q2": NumberQuestion()}
    totals = [
        sum(result.proportion for result in mark_group(marking, answers, questions).values())
        for answers in (
            {"Q1": "6", "Q2": "24"},
            {"Q1": "9", "Q2": "36"},
            {"Q1": "6", "Q2": "36"},
            {"Q1": "5", "Q2": "7"},
        )
    ]
    assert totals == [2, 1, 1, 0]


def test_a_condition_met_by_any_one_check_is_correct_when_one_does() -> None:
    marking = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition_met_by": "any_correct",
            "condition": [accepted_answer("a", question="Q1"), accepted_answer("z", question="Q3")],
            "then": [accepted_answer("x", question="Q2")],
        },
    }
    questions = {"Q1": TextQuestion(), "Q2": TextQuestion(), "Q3": TextQuestion()}
    when_condition_true = mark_group(marking, {"Q1": "b", "Q2": "x", "Q3": "z"}, questions)
    when_condition_false = mark_group(marking, {"Q1": "b", "Q2": "x", "Q3": "y"}, questions)
    assert (
        when_condition_true["Q2"].proportion,
        when_condition_false["Q2"].proportion,
        when_condition_false["Q2"].feedback,
    ) == (1, 0, "")


def test_list_settings_need_an_item_unless_they_may_be_empty() -> None:
    with pytest.raises(InvalidError) as exc_info:
        mark_scheme_of({"uses": "gradeflow/conditional@1", "with": {"condition": []}})
    assert [(problem.field, problem.message) for problem in exc_info.value.problems] == [
        ("/with/condition", "Add at least one."),
        ("/with/then", "Required."),
    ]
    without_otherwise = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [{"question": "Q1", "uses": "gradeflow/full-marks@1"}],
            "then": [{"question": "Q1", "uses": "gradeflow/full-marks@1"}],
        },
    }
    assert mark_scheme_of(without_otherwise).markings
