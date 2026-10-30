import pytest

from gradeflow_core import InvalidError
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

from .helpers import accepted, describe, mark_all, mark_group, problems_of, proportions, scheme


def test_not_allowed_gives_the_marks_its_check_doesnt() -> None:
    check = {
        "uses": "gradeflow/not-allowed@1",
        "with": {"check": {"uses": "gradeflow/key-point@1", "with": {"point": "denser"}}},
    }
    correct, failed = mark_all(check, ["less dense", "denser"])
    assert (correct.proportion, failed.proportion) == (1, 0)
    assert (correct.feedback, failed.feedback) == ("Correct.", "Not allowed. Includes “denser”.")
    assert describe(check) == "**Not allowed:** Includes `denser`."


def test_conditional_describes_the_checks_of_each_branch_by_question() -> None:
    marking = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [accepted("a", question="Q1")],
            "then": [accepted("Paris", question="Q2")],
            "otherwise": [accepted("Rome", question="Q2")],
        },
    }
    [conditional] = scheme(marking).markings
    assert conditional.describe() == (
        "**If all are correct**\n- **Q1:** Matches `a`.\n\n"
        "**Then**\n- **Q2:** Matches `Paris`.\n\n"
        "**Else**\n- **Q2:** Matches `Rome`."
    )


def test_multiple_values_checks_each_value_in_order_or_in_any_order() -> None:
    checks = [
        {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["2"]}},
        {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["-3"]}},
    ]
    roots = ValuesQuestion(value_types=["number", "number"])
    in_order = {
        "uses": "gradeflow/multiple-values@1",
        "with": {"combine": "proportional", "checks": checks},
    }
    any_order = {
        "uses": "gradeflow/multiple-values@1",
        "with": {"combine": "proportional", "any_order": True, "checks": checks},
    }
    assert proportions(mark_all(in_order, ["2, -3", "-3, 2", "2, 5"], roots)) == [1, 0, 0.5]
    assert proportions(mark_all(any_order, ["2, -3", "-3, 2", "2, 5"], roots)) == [1, 1, 0.5]
    assert [result.feedback for result in mark_all(in_order, ["2, -3", "2, 5"], roots)] == [
        "Correct.",
        "Value 1: Correct.\nValue 2: Expected -3.",
    ]
    assert describe(in_order) == (
        "**Each one counts**\n\n- **Value 1:** Matches `2`.\n- **Value 2:** Matches `-3`."
    )
    assert problems_of(
        {
            "uses": "gradeflow/multiple-values@1",
            "with": {"checks": [checks[0], {**checks[1], "value": 2}]},
        },
        roots,
    ) == [("with.checks.1", ("value",), "Its place in the list sets the value it checks.")]


def test_checks_in_a_check_of_one_value_read_it_unless_they_name_another() -> None:
    accepted = {"uses": "gradeflow/accepted-answers@1"}
    heights = {
        "uses": "gradeflow/combination@1",
        "with": {
            "checks": [
                {
                    "id": "k2",
                    "value": 1,
                    "read_only": True,
                    **accepted,
                    "with": {"answers": ["K2"]},
                },
                {"if": "{{ checks.k2.correct }}", **accepted, "with": {"answers": ["8611"]}},
                {"if": "{{ not checks.k2.correct }}", **accepted, "with": {"answers": ["8849"]}},
            ]
        },
    }
    mountain = {"uses": "gradeflow/key-points@1", "with": {"points": ["Everest"]}}
    check = {
        "uses": "gradeflow/multiple-values@1",
        "with": {"combine": "proportional", "checks": [mountain, heights]},
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
    assert results[0]["Q2"].method == "Conditional"


def test_answer_sets_count_the_set_giving_the_most_marks() -> None:
    marking = {
        "uses": "gradeflow/answer-sets@1",
        "with": {
            "sets": [
                {
                    "name": "Expected",
                    "uses": "gradeflow/combination@1",
                    "with": {
                        "checks": [accepted("6", question="Q1"), accepted("24", question="Q2")]
                    },
                },
                {
                    "name": "Read as 9",
                    "weight": 0.5,
                    "uses": "gradeflow/combination@1",
                    "with": {
                        "checks": [accepted("9", question="Q1"), accepted("36", question="Q2")]
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
            "combine": "any",
            "condition": [accepted("a", question="Q1"), accepted("z", question="Q3")],
            "then": [accepted("x", question="Q2")],
        },
    }
    questions = {"Q1": TextQuestion(), "Q2": TextQuestion(), "Q3": TextQuestion()}
    met = mark_group(marking, {"Q1": "b", "Q2": "x", "Q3": "z"}, questions)
    unmet = mark_group(marking, {"Q1": "b", "Q2": "x", "Q3": "y"}, questions)
    assert (met["Q2"].proportion, unmet["Q2"].proportion, unmet["Q2"].feedback) == (1, 0, "")


def test_list_inputs_need_an_item_unless_they_may_be_empty() -> None:
    with pytest.raises(InvalidError) as caught:
        scheme({"uses": "gradeflow/conditional@1", "with": {"condition": []}})
    assert [(p.field, p.message) for p in caught.value.problems] == [
        (("condition",), "Add at least one."),
        (("then",), "Required."),
    ]
    otherwise_empty = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [{"question": "Q1", "uses": "gradeflow/full-marks@1"}],
            "then": [{"question": "Q1", "uses": "gradeflow/full-marks@1"}],
        },
    }
    assert scheme(otherwise_empty).markings
