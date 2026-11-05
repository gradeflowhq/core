import re
from collections.abc import Mapping
from random import Random
from typing import Any

import pytest

from gradeflow_core import Question
from gradeflow_core.editing import Decision, EditingError, can_mark_by_example
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

from ..methods.helpers import describe
from .helpers import ACCEPTS_9_81, ANSWERS, answer_proportions, check_of, marked_by_example


def decide_again(
    marking: Mapping[str, Any],
    answers: list[str],
    action: Decision,
    proportion: float = 0.5,
    *,
    question: Question | None = None,
) -> dict[str, Any]:
    """Cancel the answers' decision, if any, and decide them again."""
    cancelled = marked_by_example(marking, answers, "cancel", question=question)
    return marked_by_example(cancelled, answers, action, proportion, question=question)


def test_full_marks_accept_the_answers() -> None:
    marking = marked_by_example(ACCEPTS_9_81, ["9.8"], "full_marks")
    assert marking == {
        "question": "Q1",
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": ["9.81", "9.8"]},
    }
    assert answer_proportions(marking) == [1, 0, 1, 0]


def test_partial_marks_add_an_alternative_and_join_one_of_the_same_proportion() -> None:
    after_one_part = marked_by_example(ACCEPTS_9_81, ["-9.81"], "partial_marks", 0.5)
    after_two_parts = marked_by_example(after_one_part, ["10"], "partial_marks", 0.5)
    assert answer_proportions(after_two_parts) == [1, 0.5, 0, 0.5]
    assert len(after_two_parts["with"]["checks"]) == 2, (
        "alternatives of the same proportion are one check"
    )
    assert after_two_parts["question"] == "Q1"


def test_cancelling_stops_accepting_them_and_not_allowed_rules_them_out() -> None:
    accepts_two = marked_by_example(ACCEPTS_9_81, ["9.8"], "full_marks")
    assert answer_proportions(marked_by_example(accepts_two, ["9.8"], "cancel")) == [1, 0, 0, 0]
    part_for_ten = marked_by_example(ACCEPTS_9_81, ["10"], "partial_marks", 0.5)
    assert answer_proportions(decide_again(part_for_ten, ["10"], "not_allowed")) == [1, 0, 0, 0]


@pytest.mark.parametrize("first", ["full_marks", "partial_marks", "not_allowed"])
@pytest.mark.parametrize("then", ["full_marks", "partial_marks", "not_allowed"])
def test_an_answer_with_a_decision_is_cancelled_before_another(
    first: Decision, then: Decision
) -> None:
    with_decision = marked_by_example(ACCEPTS_9_81, ["10"], first)
    with pytest.raises(EditingError, match=re.escape("These answers already have a decision.")):
        marked_by_example(with_decision, ["9.8", "10"], then)
    if first != then:
        with pytest.raises(EditingError, match=re.escape("These answers already have a decision.")):
            marked_by_example(with_decision, ["10"], then)
    if then != "full_marks":
        with pytest.raises(EditingError, match=re.escape("These answers already have a decision.")):
            marked_by_example(ACCEPTS_9_81, ["9.810"], then, question=NumberQuestion())


@pytest.mark.parametrize("decision", ["full_marks", "partial_marks", "not_allowed"])
def test_giving_answers_the_decision_they_have_changes_nothing(decision: Decision) -> None:
    with_decision = marked_by_example(ACCEPTS_9_81, ["10"], decision)
    assert marked_by_example(with_decision, ["10"], decision) == with_decision
    assert (
        marked_by_example(ACCEPTS_9_81, ["9.810"], "full_marks", question=NumberQuestion())
        == ACCEPTS_9_81
    )


def test_questions_whose_markings_accept_answers_can_be_marked_by_example() -> None:
    questions: list[Question] = [
        TextQuestion(),
        NumberQuestion(),
        ChoiceQuestion(options=frozenset({"a"})),
        ValuesQuestion(value_types=["text", "text"]),
    ]
    assert [can_mark_by_example(question) for question in questions] == [
        True,
        True,
        False,
        False,
    ]
    with pytest.raises(EditingError):
        marked_by_example(ACCEPTS_9_81, ["a"], "full_marks", question=questions[2])


def combination_depth(node: Any) -> int:
    if isinstance(node, dict):
        return int(node.get("uses") == "gradeflow/combination@1") + max(
            (combination_depth(value) for value in node.values()), default=0
        )
    if isinstance(node, list):
        return max((combination_depth(value) for value in node), default=0)
    return 0


def test_changing_an_answer_groups_marks_replaces_its_previous_marks() -> None:
    marking = ACCEPTS_9_81
    decisions: list[tuple[Decision, float]] = [
        ("partial_marks", 0.5),
        ("full_marks", 1),
        ("partial_marks", 0.5),
        ("not_allowed", 0),
        ("full_marks", 1),
        ("not_allowed", 0),
        ("partial_marks", 0.5),
    ]
    for action, proportion in decisions * 8:
        marking = decide_again(marking, ["10"], action)
        assert answer_proportions(marking) == [1, 0, 0, proportion], action
        assert combination_depth(marking) <= 2, marking


def test_partial_marks_override_an_existing_matching_method() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/number-range@1", "with": {"min_value": "0"}}
    decisions: list[tuple[Decision, list[float]]] = [
        ("partial_marks", [1, 0, 1, 0.5]),
        ("full_marks", [1, 0, 1, 1]),
        ("not_allowed", [1, 0, 1, 0]),
        ("partial_marks", [1, 0, 1, 0.5]),
    ]
    for action, expected in decisions:
        marking = decide_again(marking, ["10"], action)
        assert answer_proportions(marking) == expected
        assert combination_depth(marking) <= 2


def test_cancelling_the_last_accepted_answer_gives_no_marks_to_every_answer() -> None:
    marking = marked_by_example(ACCEPTS_9_81, ["9.81"], "cancel")
    assert marking == {
        "question": "Q1",
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": []},
    }
    assert answer_proportions(marking) == [0, 0, 0, 0]
    assert answer_proportions(marked_by_example(marking, ["10"], "partial_marks")) == [0, 0, 0, 0.5]


@pytest.mark.parametrize(
    ("action", "proportion"), [("full_marks", 1), ("partial_marks", 0.5), ("not_allowed", 0)]
)
def test_marking_an_empty_draft_by_example_produces_valid_marking(
    action: Decision, proportion: float
) -> None:
    empty_marking = {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {}}
    marking = marked_by_example(empty_marking, ["10"], action)
    assert answer_proportions(marking) == [0, 0, 0, proportion]


@pytest.mark.parametrize("settings", [{}, {"answers": ["10"]}])
def test_not_allowed_keeps_the_answer_when_nothing_is_accepted(settings: dict[str, Any]) -> None:
    empty_marking = {
        "question": "Q1",
        "uses": "gradeflow/accepted-answers@1",
        "with": settings,
    }
    not_allowed_marking = decide_again(empty_marking, ["10"], "not_allowed")
    description = describe(check_of(not_allowed_marking))
    assert "**Not allowed:** Matches `10`." in description
    assert "- Full marks for every answer" not in description
    assert answer_proportions(not_allowed_marking) == [0, 0, 0, 0]

    accepting_marking = marked_by_example(not_allowed_marking, ["9.81"], "full_marks")
    description = describe(check_of(accepting_marking))
    assert "**Not allowed:** Matches `10`." in description
    assert answer_proportions(accepting_marking) == [1, 0, 0, 0]
    assert answer_proportions(decide_again(accepting_marking, ["10"], "full_marks")) == [1, 0, 0, 1]


def test_repeated_rejections_keep_each_answer() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {}}
    for answer in ["10", "9.8", "-9.81"]:
        marking = marked_by_example(marking, [answer], "not_allowed")
    description = describe(check_of(marking))
    assert all(f"`{answer}`" in description for answer in ["10", "9.8", "-9.81"])
    assert answer_proportions(marking) == [0, 0, 0, 0]


@pytest.mark.parametrize("action", ["full_marks", "partial_marks"])
def test_new_accepted_answers_share_the_existing_not_allowed_check(action: Decision) -> None:
    empty_marking = {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {}}
    not_allowed_marking = marked_by_example(empty_marking, ["10"], "not_allowed")
    marking = marked_by_example(not_allowed_marking, ["9.81"], action)
    assert marking["with"]["combine"] == "all"
    not_allowed_check, accepting_checks = marking["with"]["checks"]
    assert not_allowed_check == {
        "uses": "gradeflow/not-allowed@1",
        "with": {"check": {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["10"]}}},
    }
    if action == "full_marks":
        assert accepting_checks == {
            "uses": "gradeflow/accepted-answers@1",
            "with": {"answers": ["9.81"]},
        }
    else:
        assert accepting_checks == {
            "uses": "gradeflow/combination@1",
            "with": {
                "combine": "any",
                "checks": [
                    {
                        "weight": 0.5,
                        "uses": "gradeflow/accepted-answers@1",
                        "with": {"answers": ["9.81"]},
                    }
                ],
            },
        }
    assert answer_proportions(marking) == [1 if action == "full_marks" else 0.5, 0, 0, 0]


def test_full_and_partial_answers_stay_together_beside_the_not_allowed_check() -> None:
    marking = marked_by_example(ACCEPTS_9_81, ["10"], "not_allowed")
    actions: list[Decision] = ["partial_marks", "full_marks", "partial_marks"]
    for action in actions:
        marking = decide_again(marking, ["9.8"], action)
        assert marking["with"]["combine"] == "all"
        not_allowed_check, accepting_checks = marking["with"]["checks"]
        assert not_allowed_check["uses"] == "gradeflow/not-allowed@1"
        assert combination_depth(marking) == (1 if action == "full_marks" else 2)
        assert answer_proportions(marking) == [1, 0, 1 if action == "full_marks" else 0.5, 0]
        if action == "partial_marks":
            assert accepting_checks["with"]["combine"] == "any"
            assert all(
                check["uses"] == "gradeflow/accepted-answers@1"
                for check in accepting_checks["with"]["checks"]
            )


def test_existing_nested_accepted_answers_are_flattened_on_the_next_edit() -> None:
    rejected = marked_by_example(ACCEPTS_9_81, ["10"], "not_allowed")
    nested = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "any",
            "checks": [
                check_of(rejected),
                {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["-9.81"]}},
            ],
        },
    }
    marking = marked_by_example(nested, ["9.8"], "full_marks")
    assert marking["with"]["combine"] == "all"
    assert combination_depth(marking) == 1
    assert answer_proportions(marking) == [1, 1, 1, 0]


@pytest.mark.parametrize(
    "other_settings",
    [
        {"answers": ["10"]},
        {"answers": ["1e1"]},
        {"answers": ["9.99"], "tolerance": 0.02},
    ],
)
def test_a_not_allowed_check_stays_local_when_another_alternative_accepts_its_answer(
    other_settings: dict[str, Any],
) -> None:
    rejected = marked_by_example(ACCEPTS_9_81, ["10"], "not_allowed")
    marking = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "any",
            "checks": [
                check_of(rejected),
                {"uses": "gradeflow/accepted-answers@1", "with": other_settings},
            ],
        },
    }
    assert answer_proportions(
        marked_by_example(marking, ["9.8"], "partial_marks", question=NumberQuestion())
    ) == [
        1,
        0,
        0.5,
        1,
    ]


def test_an_empty_combination_does_not_award_other_answers_full_marks() -> None:
    empty = {"question": "Q1", "uses": "gradeflow/combination@1", "with": {"checks": []}}
    assert answer_proportions(marked_by_example(empty, ["10"], "partial_marks")) == [0, 0, 0, 0.5]


def test_cancelling_the_answers_of_a_weighted_check_keeps_its_weight() -> None:
    marking = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "proportional",
            "checks": [
                {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["9.81"]}},
                {
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": ["10"]},
                    "weight": 0.25,
                },
            ],
        },
    }
    assert answer_proportions(marking) == [0.8, 0, 0, 0.2]
    assert answer_proportions(marked_by_example(marking, ["10"], "cancel")) == [0.8, 0, 0, 0]


def test_changing_an_answer_preserves_other_answers_in_conjunctions() -> None:
    marking = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "checks": [
                {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["9.81"]}},
                {"uses": "gradeflow/number-range@1", "with": {"min_value": 0}},
            ]
        },
    }
    assert answer_proportions(decide_again(marking, ["9.81"], "partial_marks")) == [0.5, 0, 0, 0]


def test_changing_one_example_keeps_the_other_answer_groups_decisions() -> None:
    marking = ACCEPTS_9_81
    marking = marked_by_example(marking, ["-9.81"], "partial_marks")
    marking = marked_by_example(marking, ["9.8"], "full_marks")
    decisions: list[tuple[Decision, float]] = [
        ("partial_marks", 0.5),
        ("not_allowed", 0),
        ("full_marks", 1),
    ]
    for action, proportion in decisions * 8:
        marking = decide_again(marking, ["10"], action)
        assert answer_proportions(marking) == [1, 0.5, 1, proportion]
        assert combination_depth(marking) <= 3


def test_mixed_decisions_across_answer_groups_do_not_accumulate_history() -> None:
    generator = Random(42)
    marking = ACCEPTS_9_81
    expected = [1.0, 0.0, 0.0, 0.0]
    actions: list[Decision] = ["full_marks", "partial_marks", "not_allowed"]
    for _ in range(200):
        index, action = generator.randrange(4), generator.choice(actions)
        marking = decide_again(marking, [ANSWERS[index]], action)
        expected[index] = {"full_marks": 1.0, "partial_marks": 0.5, "not_allowed": 0.0}[action]
        assert answer_proportions(marking) == expected
        assert combination_depth(marking) <= 4


def test_equivalent_answers_share_a_decision() -> None:
    question = NumberQuestion()
    marking = decide_again(ACCEPTS_9_81, ["9.810"], "partial_marks", question=question)
    assert answer_proportions(marking) == [0.5, 0, 0, 0]
    marking = marked_by_example(marking, ["9.81"], "cancel", question=question)
    assert answer_proportions(marking) == [0, 0, 0, 0]


def test_an_exact_decision_keeps_the_other_answers_a_tolerance_accepts() -> None:
    marking = {
        "question": "Q1",
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": ["9.8"], "tolerance": 0.02},
    }
    assert answer_proportions(marking) == [1, 0, 1, 0]
    changed = marked_by_example(marking, ["9.81"], "partial_marks", question=NumberQuestion())
    assert answer_proportions(changed) == [0.5, 0, 1, 0]
    assert marking["with"] == {"answers": ["9.8"], "tolerance": 0.02}


def test_existing_nested_decisions_are_simplified_when_replaced() -> None:
    marking = {
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": ["9.81", "10"]},
    }
    for _ in range(8):
        marking = {
            "uses": "gradeflow/combination@1",
            "with": {
                "checks": [
                    marking,
                    {
                        "uses": "gradeflow/not-allowed@1",
                        "with": {
                            "check": {
                                "uses": "gradeflow/accepted-answers@1",
                                "with": {"answers": ["10"]},
                            }
                        },
                    },
                ]
            },
        }
        marking = {
            "uses": "gradeflow/combination@1",
            "with": {
                "combine": "any",
                "checks": [
                    marking,
                    {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["10"]}},
                ],
            },
        }
    marking["question"] = "Q1"
    changed = decide_again(marking, ["10"], "partial_marks")
    assert answer_proportions(changed) == [1, 0, 0, 0.5]
    assert combination_depth(changed) <= 2


def test_examples_validate_the_proportion_and_leave_empty_selections_unchanged() -> None:
    with pytest.raises(EditingError, match="between 0 and 1"):
        marked_by_example(ACCEPTS_9_81, ["10"], "partial_marks", 2)
    with pytest.raises(EditingError, match="between 0 and 1"):
        marked_by_example(ACCEPTS_9_81, [], "partial_marks", 2)
    with pytest.raises(EditingError):
        marked_by_example(ACCEPTS_9_81, ["10"], "most")  # type: ignore[arg-type]
    assert marked_by_example(ACCEPTS_9_81, [], "full_marks") == ACCEPTS_9_81


def test_conditional_checks_keep_their_original_scope() -> None:
    marking: dict[str, Any] = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "checks": [
                {
                    "id": "prerequisite",
                    "read_only": True,
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": ["42"]},
                },
                {
                    "if": "{{ checks.prerequisite.correct }}",
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": ["9.81"]},
                },
                {"uses": "gradeflow/full-marks@1"},
            ]
        },
    }
    assert answer_proportions(marking) == [1, 1, 1, 1]
    for _ in range(3):
        marking = decide_again(marking, ["10"], "partial_marks")
        assert answer_proportions(marking) == [1, 1, 1, 0.5]
        marking = decide_again(marking, ["10"], "not_allowed")
        assert answer_proportions(marking) == [1, 1, 1, 0]
