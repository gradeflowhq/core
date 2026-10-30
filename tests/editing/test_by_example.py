import re
from collections.abc import Mapping
from random import Random
from typing import Any

import pytest

from gradeflow_core import Question
from gradeflow_core.editing import Decision, EditingError, can_mark_by_example, mark_by_example
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

from ..methods.helpers import describe, mark_all, proportions

ACCEPTING: dict[str, Any] = {
    "question": "Q1",
    "uses": "gradeflow/accepted-answers@1",
    "with": {"answers": ["9.81"]},
}
ANSWERS = ["9.81", "-9.81", "9.8", "10"]


def redecide(
    marking: Mapping[str, Any],
    answers: list[str],
    action: Decision,
    proportion: float = 0.5,
    *,
    question: Question | None = None,
) -> dict[str, Any]:
    """Cancel the answers' decision, if any, and decide them again."""
    cancelled = mark_by_example(marking, answers, "cancel", question=question)
    return mark_by_example(cancelled, answers, action, proportion, question=question)


def given(marking: Mapping[str, object]) -> list[float]:
    check = {key: value for key, value in marking.items() if key != "question"}
    return proportions(mark_all(check, ANSWERS, NumberQuestion()))


def test_full_marks_accept_the_answers() -> None:
    marking = mark_by_example(ACCEPTING, ["9.8"], "full")
    assert marking == {
        "question": "Q1",
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": ["9.81", "9.8"]},
    }
    assert given(marking) == [1, 0, 1, 0]


def test_part_marks_add_an_alternative_and_join_one_of_the_same_share() -> None:
    once = mark_by_example(ACCEPTING, ["-9.81"], "part", 0.5)
    twice = mark_by_example(once, ["10"], "part", 0.5)
    assert given(twice) == [1, 0.5, 0, 0.5]
    assert len(twice["with"]["checks"]) == 2, "alternatives of the same share are one check"
    assert twice["question"] == "Q1"


def test_cancelling_stops_accepting_them_and_not_allowed_rules_them_out() -> None:
    accepting_two = mark_by_example(ACCEPTING, ["9.8"], "full")
    assert given(mark_by_example(accepting_two, ["9.8"], "cancel")) == [1, 0, 0, 0]
    partly = mark_by_example(ACCEPTING, ["10"], "part", 0.5)
    assert given(redecide(partly, ["10"], "not_allowed")) == [1, 0, 0, 0]


@pytest.mark.parametrize("first", ["full", "part", "not_allowed"])
@pytest.mark.parametrize("then", ["full", "part", "not_allowed"])
def test_an_answer_with_a_decision_is_cancelled_before_another(
    first: Decision, then: Decision
) -> None:
    decided = mark_by_example(ACCEPTING, ["10"], first)
    with pytest.raises(EditingError, match=re.escape("These answers already have a decision.")):
        mark_by_example(decided, ["9.8", "10"], then)
    if first != then:
        with pytest.raises(EditingError, match=re.escape("These answers already have a decision.")):
            mark_by_example(decided, ["10"], then)
    if then != "full":
        with pytest.raises(EditingError, match=re.escape("These answers already have a decision.")):
            mark_by_example(ACCEPTING, ["9.810"], then, question=NumberQuestion())


@pytest.mark.parametrize("given", ["full", "part", "not_allowed"])
def test_giving_answers_the_decision_they_have_changes_nothing(given: Decision) -> None:
    decided = mark_by_example(ACCEPTING, ["10"], given)
    assert mark_by_example(decided, ["10"], given) == decided
    assert mark_by_example(ACCEPTING, ["9.810"], "full", question=NumberQuestion()) == ACCEPTING


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
        mark_by_example(ACCEPTING, ["a"], "full", question=questions[2])


def depth(node: Any) -> int:
    if isinstance(node, dict):
        return int(node.get("uses") == "gradeflow/combination@1") + max(
            (depth(value) for value in node.values()), default=0
        )
    if isinstance(node, list):
        return max((depth(value) for value in node), default=0)
    return 0


def test_changing_a_groups_marks_replaces_its_previous_marks() -> None:
    marking = ACCEPTING
    decisions: list[tuple[Decision, float]] = [
        ("part", 0.5),
        ("full", 1),
        ("part", 0.5),
        ("not_allowed", 0),
        ("full", 1),
        ("not_allowed", 0),
        ("part", 0.5),
    ]
    for action, proportion in decisions * 8:
        marking = redecide(marking, ["10"], action)
        assert given(marking) == [1, 0, 0, proportion], action
        assert depth(marking) <= 2, marking


def test_half_marks_override_an_existing_matching_method() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/number-range@1", "with": {"min_value": "0"}}
    decisions: list[tuple[Decision, list[float]]] = [
        ("part", [1, 0, 1, 0.5]),
        ("full", [1, 0, 1, 1]),
        ("not_allowed", [1, 0, 1, 0]),
        ("part", [1, 0, 1, 0.5]),
    ]
    for action, expected in decisions:
        marking = redecide(marking, ["10"], action)
        assert given(marking) == expected
        assert depth(marking) <= 2


def test_cancelling_the_last_accepted_answer_gives_no_marks_to_every_answer() -> None:
    marking = mark_by_example(ACCEPTING, ["9.81"], "cancel")
    assert marking == {
        "question": "Q1",
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": []},
    }
    assert given(marking) == [0, 0, 0, 0]
    assert given(mark_by_example(marking, ["10"], "part")) == [0, 0, 0, 0.5]


def test_marking_an_empty_draft_by_example_produces_valid_marking() -> None:
    empty = {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {}}
    decisions: list[tuple[Decision, float]] = [("full", 1), ("part", 0.5), ("not_allowed", 0)]
    for action, expected in decisions:
        assert given(mark_by_example(empty, ["10"], action)) == [0, 0, 0, expected]


@pytest.mark.parametrize("settings", [{}, {"answers": ["10"]}])
def test_not_allowed_keeps_the_answer_when_nothing_is_accepted(settings: dict[str, Any]) -> None:
    empty = {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": settings}
    rejected = redecide(empty, ["10"], "not_allowed")
    summary = describe({key: value for key, value in rejected.items() if key != "question"})
    assert "**Not allowed:** Matches `10`." in summary
    assert "- Full marks for every answer" not in summary
    assert given(rejected) == [0, 0, 0, 0]

    accepted = mark_by_example(rejected, ["9.81"], "full")
    summary = describe({key: value for key, value in accepted.items() if key != "question"})
    assert "**Not allowed:** Matches `10`." in summary
    assert given(accepted) == [1, 0, 0, 0]
    assert given(redecide(accepted, ["10"], "full")) == [1, 0, 0, 1]


def test_repeated_rejections_keep_each_answer() -> None:
    marking = {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {}}
    for answer in ["10", "9.8", "-9.81"]:
        marking = mark_by_example(marking, [answer], "not_allowed")
    summary = describe({key: value for key, value in marking.items() if key != "question"})
    assert all(f"`{answer}`" in summary for answer in ["10", "9.8", "-9.81"])
    assert given(marking) == [0, 0, 0, 0]


@pytest.mark.parametrize("action", ["full", "part"])
def test_new_awards_share_the_existing_exclusion(action: Decision) -> None:
    empty = {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {}}
    rejected = mark_by_example(empty, ["10"], "not_allowed")
    marking = mark_by_example(rejected, ["9.81"], action)
    assert marking["with"]["combine"] == "all"
    exclusion, awards = marking["with"]["checks"]
    assert exclusion == {
        "uses": "gradeflow/not-allowed@1",
        "with": {"check": {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["10"]}}},
    }
    if action == "full":
        assert awards == {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["9.81"]}}
    else:
        assert awards == {
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
    assert given(marking) == [1 if action == "full" else 0.5, 0, 0, 0]


def test_full_and_partial_answers_stay_together_beside_the_exclusion() -> None:
    marking = mark_by_example(ACCEPTING, ["10"], "not_allowed")
    actions: list[Decision] = ["part", "full", "part"]
    for action in actions:
        marking = redecide(marking, ["9.8"], action)
        assert marking["with"]["combine"] == "all"
        exclusion, awards = marking["with"]["checks"]
        assert exclusion["uses"] == "gradeflow/not-allowed@1"
        assert depth(marking) == (1 if action == "full" else 2)
        assert given(marking) == [1, 0, 1 if action == "full" else 0.5, 0]
        if action == "part":
            assert awards["with"]["combine"] == "any"
            assert all(
                check["uses"] == "gradeflow/accepted-answers@1"
                for check in awards["with"]["checks"]
            )


def test_existing_nested_awards_are_factored_on_the_next_edit() -> None:
    rejected = mark_by_example(ACCEPTING, ["10"], "not_allowed")
    nested = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "any",
            "checks": [
                {key: value for key, value in rejected.items() if key != "question"},
                {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["-9.81"]}},
            ],
        },
    }
    marking = mark_by_example(nested, ["9.8"], "full")
    assert marking["with"]["combine"] == "all"
    assert depth(marking) == 1
    assert given(marking) == [1, 1, 1, 0]


@pytest.mark.parametrize(
    "alternative",
    [
        {"answers": ["10"]},
        {"answers": ["1e1"]},
        {"answers": ["9.99"], "tolerance": 0.02},
    ],
)
def test_an_exclusion_stays_local_when_another_alternative_accepts_its_answer(
    alternative: dict[str, Any],
) -> None:
    rejected = mark_by_example(ACCEPTING, ["10"], "not_allowed")
    marking = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "any",
            "checks": [
                {key: value for key, value in rejected.items() if key != "question"},
                {"uses": "gradeflow/accepted-answers@1", "with": alternative},
            ],
        },
    }
    assert given(mark_by_example(marking, ["9.8"], "part", question=NumberQuestion())) == [
        1,
        0,
        0.5,
        1,
    ]


def test_an_empty_combination_does_not_award_other_answers_full_marks() -> None:
    empty = {"question": "Q1", "uses": "gradeflow/combination@1", "with": {"checks": []}}
    assert given(mark_by_example(empty, ["10"], "part")) == [0, 0, 0, 0.5]


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
    assert given(redecide(marking, ["9.81"], "part")) == [0.5, 0, 0, 0]


def test_changing_one_example_keeps_the_other_groups_decisions() -> None:
    marking = ACCEPTING
    marking = mark_by_example(marking, ["-9.81"], "part")
    marking = mark_by_example(marking, ["9.8"], "full")
    decisions: list[tuple[Decision, float]] = [("part", 0.5), ("not_allowed", 0), ("full", 1)]
    for action, proportion in decisions * 8:
        marking = redecide(marking, ["10"], action)
        assert given(marking) == [1, 0.5, 1, proportion]
        assert depth(marking) <= 3


def test_mixed_decisions_across_groups_do_not_accumulate_history() -> None:
    random = Random(42)
    marking = ACCEPTING
    expected = [1.0, 0.0, 0.0, 0.0]
    actions: list[Decision] = ["full", "part", "not_allowed"]
    for _ in range(200):
        index, action = random.randrange(4), random.choice(actions)
        marking = redecide(marking, [ANSWERS[index]], action)
        expected[index] = {"full": 1.0, "part": 0.5, "not_allowed": 0.0}[action]
        assert given(marking) == expected
        assert depth(marking) <= 4


def test_equivalent_answers_share_a_decision() -> None:
    question = NumberQuestion()
    marking = redecide(ACCEPTING, ["9.810"], "part", question=question)
    assert given(marking) == [0.5, 0, 0, 0]
    marking = mark_by_example(marking, ["9.81"], "cancel", question=question)
    assert given(marking) == [0, 0, 0, 0]


def test_an_exact_decision_keeps_the_other_answers_a_tolerance_accepts() -> None:
    marking = {
        "question": "Q1",
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": ["9.8"], "tolerance": 0.02},
    }
    assert given(marking) == [1, 0, 1, 0]
    changed = mark_by_example(marking, ["9.81"], "part", question=NumberQuestion())
    assert given(changed) == [0.5, 0, 1, 0]
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
    changed = redecide(marking, ["10"], "part")
    assert given(changed) == [1, 0, 0, 0.5]
    assert depth(changed) <= 2


def test_examples_validate_the_proportion_and_leave_empty_selections_unchanged() -> None:
    with pytest.raises(EditingError, match="between 0 and 1"):
        mark_by_example(ACCEPTING, ["10"], "part", 2)
    with pytest.raises(EditingError):
        mark_by_example(ACCEPTING, ["10"], "most")  # type: ignore[arg-type]
    assert mark_by_example(ACCEPTING, [], "full") == ACCEPTING


def test_conditional_checks_keep_their_original_scope() -> None:
    marking: dict[str, Any] = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "checks": [
                {
                    "id": "gate",
                    "read_only": True,
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": ["42"]},
                },
                {
                    "if": "{{ checks.gate.correct }}",
                    "uses": "gradeflow/accepted-answers@1",
                    "with": {"answers": ["9.81"]},
                },
                {"uses": "gradeflow/full-marks@1"},
            ]
        },
    }
    assert given(marking) == [1, 1, 1, 1]
    for _ in range(3):
        marking = redecide(marking, ["10"], "part")
        assert given(marking) == [1, 1, 1, 0.5]
        marking = redecide(marking, ["10"], "not_allowed")
        assert given(marking) == [1, 1, 1, 0]
