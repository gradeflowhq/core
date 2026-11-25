from typing import Any

import pytest

from gradeflow_core import InvalidError, QuestionSet
from gradeflow_core.editing import MarkingDraft
from gradeflow_core.questions.list import ListQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.parts import AnswerPart
from gradeflow_core.questions.reading import ListReading
from gradeflow_core.questions.text import TextQuestion

from .helpers import (
    accepted_answer,
    describe,
    mark_all,
    mark_group,
    mark_scheme_of,
    multipart,
    problems_of,
    proportions,
    with_part_inputs,
    workflow,
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


ROOT_CHECKS = [accepted_answer("2"), accepted_answer("-3")]


def item_by_item(*checks: dict[str, Any], marks_by: str = "proportional") -> dict[str, Any]:
    return {
        "uses": "gradeflow/item-by-item@1",
        "with": {"marks_by": marks_by, "checks": list(checks)},
    }


def test_part_by_part_checks_each_part_where_it_is() -> None:
    roots = multipart("number", "number")
    marking = {
        "uses": "gradeflow/part-by-part@1",
        "with": {"marks_by": "proportional", "checks": with_part_inputs(*ROOT_CHECKS)},
    }
    assert proportions(mark_all(marking, ["2, -3", "-3, 2", "2, 5"], roots)) == [1, 0, 0.5]
    assert [result.feedback for result in mark_all(marking, ["2, -3", "2, 5"], roots)] == [
        "Correct.",
        "Part 1: Correct.\nPart 2: Expected -3.",
    ]
    assert describe(marking) == (
        "Split by weight between these:\n- **Part 1:** Matches `2`.\n- **Part 2:** Matches `-3`."
    )
    assert problems_of(
        {"uses": "gradeflow/part-by-part@1", "with": {"checks": ROOT_CHECKS}}, roots
    ) == [
        (
            f"/with/checks/{index}",
            "/input",
            "Accepted Answers can't be used on the whole answer. Choose one of its parts as the "
            "input.",
        )
        for index in range(2)
    ]


def test_part_by_part_says_what_each_part_is_worth() -> None:
    roots = multipart("number", "number")
    checks = [accepted_answer("2", weight=2), accepted_answer("-3", name="Second root", weight=-1)]
    marking = {
        "uses": "gradeflow/part-by-part@1",
        "with": {"marks_by": "total", "checks": with_part_inputs(*checks)},
    }
    assert describe(marking) == (
        "Marks added up from these:\n- **Part 1** (2 marks): Matches `2`.\n"
        "- **Second root** (Penalty 1 mark): Matches `-3`."
    )
    assert [result.feedback for result in mark_all(marking, ["5, -3"], roots)] == [
        "Part 1: Expected 2.\nPenalty: Second root."
    ]


def test_part_by_part_needs_a_check_for_each_part() -> None:
    roots = multipart("number", "number")

    def problems_with(*checks: dict[str, Any]) -> list[tuple[str, str, str]]:
        marking = {
            "uses": "gradeflow/part-by-part@1",
            "with": {"checks": with_part_inputs(*checks)},
        }
        return problems_of(marking, roots)

    count_problem = ("", "/with/checks", "List 2 checks, one for each part.")
    assert problems_with(accepted_answer("2")) == [count_problem]
    assert problems_with(*ROOT_CHECKS, accepted_answer("5")) == [
        count_problem,
        ("/with/checks/2", "/input", "The answer has no part named part_3."),
    ]


def test_part_by_part_checks_its_weights_against_how_marks_are_given() -> None:
    roots = multipart("number", "number")
    checks = with_part_inputs(accepted_answer("2"), {**accepted_answer("-3"), "weight": 2})

    def problems_with(marks_by: str) -> list[tuple[str, str, str]]:
        marking = {
            "uses": "gradeflow/part-by-part@1",
            "with": {"marks_by": marks_by, "checks": checks},
        }
        return problems_of(marking, roots)

    assert problems_with("all_correct") == [
        (
            "/with/checks/1",
            "/weight",
            "“All must be correct” takes only 1 (counts) or 0 (ignored).",
        )
    ]
    assert problems_with("total") == []


def test_item_by_item_pairs_items_with_checks_in_any_order() -> None:
    roots = ListQuestion(item="number")
    marking = item_by_item(*ROOT_CHECKS)
    results = mark_all(marking, ["2, -3", "-3, 2", "2, 5", "-3"], roots)
    assert proportions(results) == [1, 1, 0.5, 0.5]
    assert [result.feedback for result in results[2:]] == [
        "Item 1: Correct.\nItem 2: Expected -3.",
        "Item 1: Correct.\nItem 2: No answer.",
    ]
    assert describe(marking) == (
        "Split by weight between these, in any order:\n- Matches `2`.\n- Matches `-3`."
    )
    assert problems_of(marking, roots) == []


def test_a_tie_pairs_checks_with_the_items_they_match() -> None:
    [result] = mark_all(item_by_item(*ROOT_CHECKS, marks_by="all_correct"), ["-3"], ListQuestion())
    assert (result.proportion, result.feedback) == (0, "Item 2: No answer.")


def test_each_extra_item_costs_an_average_checks_share() -> None:
    roots = ListQuestion(item="number")
    [proportional, total, all_correct] = [
        mark_all(item_by_item(*ROOT_CHECKS, marks_by=marks_by), ["2, -3, 5, 7"], roots)[0]
        for marks_by in ("proportional", "total", "all_correct")
    ]
    assert proportional.proportion == 0.5
    assert proportional.feedback == "Item 1: Correct.\nItem 2: Correct.\n2 extra items."
    [unordered] = mark_all(item_by_item(*ROOT_CHECKS), ["7, -3, 2"], roots)
    assert unordered.feedback == "Item 2: Correct.\nItem 3: Correct.\n1 extra item."
    assert total.proportion == 1
    assert all_correct.proportion == 0


def test_items_in_parts_stay_together() -> None:
    question = ListQuestion(
        item="parts",
        parts=[AnswerPart(name="problem"), AnswerPart(name="change")],
        reading=ListReading(separator="~"),
    )

    def problem(name: str, change: str) -> dict[str, Any]:
        return {
            "name": name.capitalize(),
            "uses": "gradeflow/part-by-part@1",
            "with": {
                "checks": [
                    {"input": "{{ answer.problem }}", **accepted_answer(name)},
                    {"input": "{{ answer.change }}", **accepted_answer(change)},
                ]
            },
        }

    marking = item_by_item(problem("overfitting", "add l2"), problem("imbalance", "use f1"))
    answers = [
        "Overfitting~add L2~Imbalance~use F1",
        "Imbalance~use F1~Overfitting~add L2",
        "Overfitting~use F1~Imbalance~add L2",
    ]
    assert proportions(mark_all(marking, answers, question)) == [1, 1, 0]
    assert problems_of(marking, question) == []


def test_item_by_item_starts_with_one_check_unless_items_are_in_parts() -> None:
    questions = QuestionSet(
        {
            "Q1": ListQuestion(),
            "Q2": ListQuestion(
                item="parts", parts=[AnswerPart(name="problem"), AnswerPart(name="change")]
            ),
        }
    )
    single = MarkingDraft.new("gradeflow/item-by-item@1", questions, "Q1").data
    assert [check["uses"] for check in single["with"]["checks"]] == ["gradeflow/accepted-answers@1"]
    in_parts = MarkingDraft.new("gradeflow/item-by-item@1", questions, "Q2").data
    assert in_parts["with"].get("checks", []) == []


def test_item_by_item_pairs_up_to_six_checks() -> None:
    marking = item_by_item(*[accepted_answer(str(number)) for number in range(7)])
    with pytest.raises(InvalidError, match="Keep at most 6"):
        problems_of(marking, ListQuestion())


def test_a_part_is_read_from_what_the_step_holding_it_reads() -> None:
    question = multipart("text", "text")
    k2 = accepted_answer("K2", id="k2", input="{{ answer.part_1 }}", weight=0)
    heights = [
        accepted_answer(
            "8611", input="{{ answer.part_2 }}", **{"if": "{{ steps.k2.output.correct }}"}
        ),
        accepted_answer(
            "8849", input="{{ answer.part_2 }}", **{"if": "{{ not steps.k2.output.correct }}"}
        ),
    ]
    mountain = {
        "input": "{{ answer.part_1 }}",
        "uses": "gradeflow/key-points@1",
        "with": {"points": ["Everest"]},
    }
    marking = workflow("proportional", mountain, k2, *heights)
    answers = ["Everest, 8849", "K2, 8611", "K2, Everest"]
    assert proportions(mark_all(marking, answers, question)) == [1, 0.5, 0]
    assert problems_of(marking, question) == []
    height_check = {**workflow("all_correct", {**k2, "weight": 1}), "input": "{{ answer.part_2 }}"}
    part_by_part = {
        "uses": "gradeflow/part-by-part@1",
        "with": {"checks": [mountain, height_check]},
    }
    assert problems_of(part_by_part, question) == [
        ("/with/checks/1/with/steps/0", "/input", "Only a multi-part answer has parts to read.")
    ]


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
