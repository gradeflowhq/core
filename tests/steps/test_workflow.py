import pytest

from gradeflow_core import Question
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from ..methods.helpers import (
    accepted_answer,
    describe,
    key_point,
    mark,
    mark_all,
    mark_group,
    problems_of,
    proportions,
    workflow,
)


@pytest.mark.parametrize(
    ("marks_by", "expected"),
    [
        ("all_correct", [1, 0, 0]),
        ("any_correct", [1, 1, 0]),
        ("none_correct", [0, 0, 1]),
        ("total", [1, 1, 0]),
        ("proportional", [1, 0.5, 0]),
    ],
)
def test_marks_by_decides_the_proportion(marks_by: str, expected: list[float]) -> None:
    check = workflow(marks_by, key_point("a"), key_point("b"))
    assert proportions(mark_all(check, ["a b", "a", "c"])) == expected


def test_weights_count_under_any_correct_and_proportional() -> None:
    alternatives = workflow(
        "any_correct",
        accepted_answer("9.81", name="Correct"),
        accepted_answer("-9.81", name="Sign error", weight=0.5),
    )
    correct, sign_error, wrong = mark_all(alternatives, ["9.81", "-9.81", "10"], NumberQuestion())
    assert (correct.proportion, sign_error.proportion, wrong.proportion) == (1, 0.5, 0)
    assert sign_error.feedback == "Sign error (50% of the marks).", "its name says why"
    assert correct.feedback == "Correct.", "not Correct: Correct."
    assert wrong.feedback == "Expected 9.81.", "no check that gave nothing speaks by name"
    workflow_of_unnamed_steps = workflow(
        "any_correct", accepted_answer("9.81"), accepted_answer("-9.81", weight=0.5)
    )
    assert (
        mark(workflow_of_unnamed_steps, "-9.81", NumberQuestion()).feedback
        == "Partly correct (50% of the marks)."
    )
    with_own_feedback = accepted_answer(
        "-9.81", name="Sign error", weight=0.5, feedback={"correct": "Check g's direction."}
    )
    [sign_error] = mark_all(workflow("any_correct", with_own_feedback), ["-9.81"], NumberQuestion())
    assert sign_error.feedback == "Check g's direction.", "as its author wrote it"
    with_penalty = workflow("proportional", key_point("a", weight=2), key_point("x", weight=-1))
    assert proportions(mark_all(with_penalty, ["a", "a x", "x"])) == [1, 0.5, 0]


def test_total_gives_weights_as_marks_capped_at_max_marks() -> None:
    total_workflow = workflow(
        "total", *(key_point(point, weight=2) for point in "abc"), key_point("x", weight=-1)
    )
    four_mark_question = TextQuestion(max_marks=4)
    results = mark_all(total_workflow, ["a", "a b", "a b c", "a b c x", "x"], four_mark_question)
    assert proportions(results) == [0.5, 1, 1, 1, 0]
    assert describe(total_workflow).startswith("Total of these:\n- ")


def test_feedback_follows_each_name_and_any_correct_gives_only_the_one_that_counted() -> None:
    all_correct_workflow = workflow("all_correct", key_point("a", name="A"), key_point("b"))
    assert mark(all_correct_workflow, "a b").feedback == "A: Has “a”.\nHas “b”."
    any_correct_workflow = workflow("any_correct", key_point("a"), key_point("b"))
    assert mark(any_correct_workflow, "b").feedback == "Has “b”."


def test_a_penalty_gives_feedback_only_when_it_applies() -> None:
    titration = workflow(
        "proportional",
        accepted_answer("25.10"),
        accepted_answer("25.40", name="Rough titre", weight=-1),
    )
    correct_result, penalised_result = mark_all(titration, ["25.10", "25.40"], NumberQuestion())
    assert correct_result.feedback == "Correct."
    assert penalised_result.feedback == "Expected 25.10.\nPenalty: Rough titre."
    with_length_penalty = workflow(
        "proportional",
        key_point("a"),
        {
            "uses": "gradeflow/not-allowed@1",
            "weight": -1,
            "with": {"check": {"uses": "gradeflow/length@1", "with": {"max_length": 1}}},
        },
    )
    within_length, over_length = mark_all(with_length_penalty, ["a", "a b"])
    assert (within_length.feedback, over_length.feedback) == ("Has “a”.", "Has “a”.\nPenalty.")


def test_at_full_marks_steps_that_gave_nothing_give_no_feedback() -> None:
    three_points_two_marks = workflow("total", *(key_point(point) for point in "abc"))
    full_marks_result, partial_marks_result = mark_all(
        three_points_two_marks, ["a c", "b"], TextQuestion(max_marks=2)
    )
    assert full_marks_result.feedback == "Has “a”.\nHas “c”."
    assert partial_marks_result.feedback == "Missing “a”.\nHas “b”.\nMissing “c”."


def test_all_must_be_correct_names_the_checks_that_werent() -> None:
    check = workflow("all_correct", key_point("a"), key_point("b"))
    assert mark(check, "a").feedback == "Missing “b”."


def test_none_may_be_correct_says_what_the_answer_did_that_is_not_allowed() -> None:
    check = workflow("none_correct", key_point("a"), key_point("b"))
    assert [result.feedback for result in mark_all(check, ["c", "a", "b a"])] == [
        "Correct.",
        "Not allowed. Includes “a”.",
        "Not allowed. Includes “a”. Includes “b”.",
    ]
    with_own_feedback = key_point("b", feedback={"correct": "Leave b out."})
    assert mark(workflow("none_correct", with_own_feedback), "b").feedback == (
        "Not allowed. Leave b out."
    )


def test_none_may_be_correct_ends_each_description_as_a_sentence() -> None:
    check = workflow("none_correct", accepted_answer("a"))
    assert mark(check, "a", TextQuestion()).feedback == "Not allowed. Matches “a”."


def test_weight_zero_checks_decide_without_marks_and_conditions_read_earlier_results() -> None:
    marking = workflow(
        "all_correct",
        accepted_answer("t-test", id="test", question="Q5", weight=0),
        accepted_answer("2.31", question="Q6", **{"if": "{{ steps.test.output.correct }}"}),
        accepted_answer("5.99", question="Q6", **{"if": "{{ not steps.test.output.correct }}"}),
    )
    questions: dict[str, Question] = {"Q5": TextQuestion(), "Q6": NumberQuestion()}
    chose_t_test = mark_group(marking, {"Q5": "t-test", "Q6": "2.31"}, questions)
    chose_chi_squared = mark_group(marking, {"Q5": "chi", "Q6": "2.31"}, questions)
    assert list(chose_t_test) == ["Q6"]
    assert (chose_t_test["Q6"].proportion, chose_chi_squared["Q6"].proportion) == (1, 0)
    assert chose_chi_squared["Q6"].feedback == "Expected 5.99."


def test_any_correct_across_questions_picks_one_check_by_its_total_marks() -> None:
    marking = workflow(
        "any_correct",
        workflow(
            "all_correct",
            accepted_answer("6", question="Q1"),
            accepted_answer("24", question="Q2"),
            name="Expected",
        ),
        workflow(
            "all_correct",
            accepted_answer("9", question="Q1"),
            accepted_answer("36", question="Q2"),
            name="Read as 9",
            weight=0.5,
        ),
    )
    questions: dict[str, Question] = {"Q1": NumberQuestion(), "Q2": NumberQuestion(max_marks=2)}
    follow_through = mark_group(marking, {"Q1": "9", "Q2": "36"}, questions)
    assert {question_id: result.proportion for question_id, result in follow_through.items()} == {
        "Q1": 0.5,
        "Q2": 0.5,
    }
    mixed_answers_results = mark_group(marking, {"Q1": "6", "Q2": "36"}, questions)
    assert {
        question_id: result.proportion for question_id, result in mixed_answers_results.items()
    } == {
        "Q1": 1,
        "Q2": 0,
    }


def test_a_workflow_describes_its_steps_with_their_names_and_weights() -> None:
    check = workflow("any_correct", key_point("a", name="A"), key_point("b", weight=0.5))
    assert describe(check) == (
        "Any one of these:\n- **A:** Includes `a`.\n- **50% of the marks:** Includes `b`."
    )
    proportional = workflow(
        "proportional",
        key_point("a"),
        key_point("b", weight=2),
        key_point("c", name="C", weight=-1),
    )
    assert describe(proportional) == (
        "Each one counts:\n"
        "- Includes `a`.\n"
        "- **Weight 2:** Includes `b`.\n"
        "- **C** (weight -1): Includes `c`."
    )


def test_a_nested_workflow_is_described_as_an_indented_list() -> None:
    check = workflow("all_correct", workflow("any_correct", key_point("a")), key_point("b"))
    assert describe(workflow("none_correct", key_point("a"))) == "None of these:\n- Includes `a`."
    assert describe(check) == (
        "All of these:\n- Any one of these:\n  - Includes `a`.\n- Includes `b`."
    )


def test_a_workflow_needs_a_step() -> None:
    assert problems_of(workflow("all_correct")) == [("", ("steps",), "Add at least one step.")]


@pytest.mark.parametrize("marks_by", ["all_correct", "none_correct"])
def test_a_weight_where_weights_arent_used_is_a_problem(marks_by: str) -> None:
    check = workflow(marks_by, key_point("a"), key_point("b", weight=0.5))
    label = {"all_correct": "All must be correct", "none_correct": "None may be correct"}[marks_by]
    assert problems_of(check) == [("/with/steps/1", ("weight",), f"“{label}” doesn't use weights.")]
    assert problems_of({**check, "with": {**check["with"], "marks_by": "any_correct"}}) == []


def test_a_step_left_unmarked_says_why_even_when_it_would_be_a_penalty() -> None:
    failing_step = {
        "uses": "gradeflow/custom-code@1",
        "with": {"code": "raise ValueError('bad')"},
        "weight": -1,
    }
    result = mark(workflow("proportional", key_point("a"), failing_step), "a")
    assert not result.is_marked
    assert "Your code failed: ValueError: bad." in result.feedback


def test_a_step_left_unmarked_says_why_when_none_may_be_correct() -> None:
    failing_step = {"uses": "gradeflow/custom-code@1", "with": {"code": "raise ValueError('bad')"}}
    result = mark(workflow("none_correct", key_point("a"), failing_step), "c")
    assert not result.is_marked
    assert "Your code failed: ValueError: bad." in result.feedback
    assert "Correct." not in result.feedback


def test_any_correct_is_left_unmarked_while_an_unmarked_check_could_give_more() -> None:
    failing_step = {"uses": "gradeflow/custom-code@1", "with": {"code": "raise ValueError('bad')"}}
    unmarked_first = mark(workflow("any_correct", failing_step, key_point("a", weight=0.5)), "a")
    assert (unmarked_first.proportion, unmarked_first.is_marked) == (0.5, False)
    assert "Your code failed: ValueError: bad." in unmarked_first.feedback
    worth_less = mark(workflow("any_correct", key_point("a"), {**failing_step, "weight": 0.5}), "a")
    assert (worth_less.proportion, worth_less.is_marked) == (1, True)
    assert worth_less.feedback == "Has “a”."


@pytest.mark.parametrize("marks_by", ["total", "proportional"])
def test_a_penalty_that_cancels_the_marks_leaves_no_marks_despite_rounding(marks_by: str) -> None:
    workflow_with_cancelling_penalty = workflow(
        marks_by,
        key_point("a", weight=0.1),
        key_point("b", weight=0.2),
        key_point("c", weight=-0.3),
    )
    result = mark(workflow_with_cancelling_penalty, "a b c")
    assert result.breakdown is not None
    assert (result.proportion, result.breakdown.effect) == (0, "no_marks"), "0.1 + 0.2 - 0.3 is 0"
