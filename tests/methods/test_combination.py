import pytest

from gradeflow_core import Question
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from .helpers import (
    accepted_answer,
    combination,
    describe,
    key_point,
    mark,
    mark_all,
    mark_group,
    problems_of,
    proportions,
)


@pytest.mark.parametrize(
    ("combine", "expected"),
    [
        ("all", [1, 0, 0]),
        ("any", [1, 1, 0]),
        ("none", [0, 0, 1]),
        ("total", [1, 1, 0]),
        ("proportional", [1, 0.5, 0]),
    ],
)
def test_how_checks_combine(combine: str, expected: list[float]) -> None:
    check = combination(combine, key_point("a"), key_point("b"))
    assert proportions(mark_all(check, ["a b", "a", "c"])) == expected


def test_weights_count_under_any_and_each_one_counts() -> None:
    alternatives = combination(
        "any",
        accepted_answer("9.81", name="Correct"),
        accepted_answer("-9.81", name="Sign error", weight=0.5),
    )
    correct, sign_error, wrong = mark_all(alternatives, ["9.81", "-9.81", "10"], NumberQuestion())
    assert (correct.proportion, sign_error.proportion, wrong.proportion) == (1, 0.5, 0)
    assert sign_error.feedback == "Sign error (50% of the marks).", "its name says why"
    assert correct.feedback == "Correct.", "not Correct: Correct."
    assert wrong.feedback == "Expected 9.81.", "no check that gave nothing speaks by name"
    unnamed = combination("any", accepted_answer("9.81"), accepted_answer("-9.81", weight=0.5))
    assert mark(unnamed, "-9.81", NumberQuestion()).feedback == "Partly correct (50% of the marks)."
    with_own_feedback = accepted_answer(
        "-9.81", name="Sign error", weight=0.5, feedback={"correct": "Check g's direction."}
    )
    [sign_error] = mark_all(combination("any", with_own_feedback), ["-9.81"], NumberQuestion())
    assert sign_error.feedback == "Check g's direction.", "as its author wrote it"
    with_penalty = combination("proportional", key_point("a", weight=2), key_point("x", weight=-1))
    assert proportions(mark_all(with_penalty, ["a", "a x", "x"])) == [1, 0.5, 0]


def test_a_total_gives_its_weights_as_marks_up_to_the_questions() -> None:
    total = combination(
        "total", *(key_point(point, weight=2) for point in "abc"), key_point("x", weight=-1)
    )
    four_mark_question = TextQuestion(max_marks=4)
    assert proportions(
        mark_all(total, ["a", "a b", "a b c", "a b c x", "x"], four_mark_question)
    ) == [
        0.5,
        1,
        1,
        1,
        0,
    ]
    assert describe(total).startswith("Total of these:\n- ")


def test_each_check_gives_feedback_after_its_name_but_any_one_only_the_one_that_counted() -> None:
    every = combination("all", key_point("a", name="A"), key_point("b"))
    assert mark(every, "a b").feedback == "A: Has “a”.\nHas “b”."
    either = combination("any", key_point("a"), key_point("b"))
    assert mark(either, "b").feedback == "Has “b”."


def test_a_penalty_gives_feedback_only_when_it_applies() -> None:
    titration = combination(
        "proportional",
        accepted_answer("25.10"),
        accepted_answer("25.40", name="Rough titre", weight=-1),
    )
    clean, rough = mark_all(titration, ["25.10", "25.40"], NumberQuestion())
    assert clean.feedback == "Correct."
    assert rough.feedback == "Expected 25.10.\nPenalty: Rough titre."
    with_length_penalty = combination(
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


def test_once_full_marks_are_reached_what_gave_nothing_stays_silent() -> None:
    three_points_two_marks = combination("total", *(key_point(point) for point in "abc"))
    full, partial = mark_all(three_points_two_marks, ["a c", "b"], TextQuestion(max_marks=2))
    assert full.feedback == "Has “a”.\nHas “c”."
    assert partial.feedback == "Missing “a”.\nHas “b”.\nMissing “c”."


def test_all_must_be_correct_names_the_checks_that_werent() -> None:
    check = combination("all", key_point("a"), key_point("b"))
    assert mark(check, "a").feedback == "Missing “b”."


def test_none_may_be_correct_says_what_the_answer_did_that_is_not_allowed() -> None:
    check = combination("none", key_point("a"), key_point("b"))
    assert [result.feedback for result in mark_all(check, ["c", "a", "b a"])] == [
        "Correct.",
        "Not allowed. Includes “a”.",
        "Not allowed. Includes “a”. Includes “b”.",
    ]
    with_own_feedback = key_point("b", feedback={"correct": "Leave b out."})
    assert mark(combination("none", with_own_feedback), "b").feedback == (
        "Not allowed. Leave b out."
    )


def test_none_may_be_correct_ends_each_description_as_a_sentence() -> None:
    check = combination("none", accepted_answer("a"))
    assert mark(check, "a", TextQuestion()).feedback == "Not allowed. Matches “a”."


def test_read_only_checks_decide_without_marks_and_conditions_read_earlier_results() -> None:
    marking = combination(
        "all",
        accepted_answer("t-test", id="test", question="Q5", read_only=True),
        accepted_answer("2.31", question="Q6", **{"if": "{{ checks.test.correct }}"}),
        accepted_answer("5.99", question="Q6", **{"if": "{{ not checks.test.correct }}"}),
    )
    questions: dict[str, Question] = {"Q5": TextQuestion(), "Q6": NumberQuestion()}
    chose_t_test = mark_group(marking, {"Q5": "t-test", "Q6": "2.31"}, questions)
    chose_chi_squared = mark_group(marking, {"Q5": "chi", "Q6": "2.31"}, questions)
    assert list(chose_t_test) == ["Q6"]
    assert (chose_t_test["Q6"].proportion, chose_chi_squared["Q6"].proportion) == (1, 0)
    assert chose_chi_squared["Q6"].feedback == "Expected 5.99."


def test_any_one_across_questions_picks_one_check_by_its_total_marks() -> None:
    marking = combination(
        "any",
        combination(
            "all",
            accepted_answer("6", question="Q1"),
            accepted_answer("24", question="Q2"),
            name="Expected",
        ),
        combination(
            "all",
            accepted_answer("9", question="Q1"),
            accepted_answer("36", question="Q2"),
            name="Read as 9",
            weight=0.5,
        ),
    )
    questions: dict[str, Question] = {"Q1": NumberQuestion(), "Q2": NumberQuestion(max_marks=2)}
    follow_through = mark_group(marking, {"Q1": "9", "Q2": "36"}, questions)
    assert {key: result.proportion for key, result in follow_through.items()} == {
        "Q1": 0.5,
        "Q2": 0.5,
    }
    mixed = mark_group(marking, {"Q1": "6", "Q2": "36"}, questions)
    assert {key: result.proportion for key, result in mixed.items()} == {"Q1": 1, "Q2": 0}


def test_it_lists_its_checks_with_their_names_and_weights() -> None:
    check = combination("any", key_point("a", name="A"), key_point("b", weight=0.5))
    assert describe(check) == (
        "Any one of these:\n- **A:** Includes `a`.\n- **50% of the marks:** Includes `b`."
    )
    proportional = combination(
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


def test_a_combination_in_a_combination_is_listed_under_its_place() -> None:
    check = combination("all", combination("any", key_point("a")), key_point("b"))
    assert describe(combination("none", key_point("a"))) == "None of these:\n- Includes `a`."
    assert describe(check) == (
        "All of these:\n- Any one of these:\n  - Includes `a`.\n- Includes `b`."
    )


def test_a_combination_needs_a_check() -> None:
    assert problems_of(combination("all")) == [("", ("checks",), "Add at least one check.")]


@pytest.mark.parametrize("combine", ["all", "none"])
def test_a_weight_where_weights_arent_used_is_a_problem(combine: str) -> None:
    check = combination(combine, key_point("a"), key_point("b", weight=0.5))
    label = {"all": "All must be correct", "none": "None may be correct"}[combine]
    assert problems_of(check) == [("with.checks.1", ("weight",), f"“{label}” doesn't use weights.")]
    assert problems_of({**check, "with": {**check["with"], "combine": "any"}}) == []


def test_a_part_left_unmarked_says_why_even_when_it_would_be_a_penalty() -> None:
    failing = {
        "uses": "gradeflow/custom-code@1",
        "with": {"code": "raise ValueError('bad')"},
        "weight": -1,
    }
    result = mark(combination("proportional", key_point("a"), failing), "a")
    assert not result.is_marked
    assert "Your code failed: ValueError: bad." in result.feedback


def test_a_part_left_unmarked_says_why_when_none_may_be_correct() -> None:
    failing = {"uses": "gradeflow/custom-code@1", "with": {"code": "raise ValueError('bad')"}}
    result = mark(combination("none", key_point("a"), failing), "c")
    assert not result.is_marked
    assert "Your code failed: ValueError: bad." in result.feedback
    assert "Correct." not in result.feedback


def test_any_one_is_left_unmarked_while_an_unmarked_check_could_give_more() -> None:
    failing = {"uses": "gradeflow/custom-code@1", "with": {"code": "raise ValueError('bad')"}}
    unmarked_first = mark(combination("any", failing, key_point("a", weight=0.5)), "a")
    assert (unmarked_first.proportion, unmarked_first.is_marked) == (0.5, False)
    assert "Your code failed: ValueError: bad." in unmarked_first.feedback
    worth_less = mark(combination("any", key_point("a"), {**failing, "weight": 0.5}), "a")
    assert (worth_less.proportion, worth_less.is_marked) == (1, True)
    assert worth_less.feedback == "Has “a”."


@pytest.mark.parametrize("combine", ["total", "proportional"])
def test_a_penalty_that_cancels_the_marks_leaves_no_marks_despite_rounding(combine: str) -> None:
    weights = combination(
        combine, key_point("a", weight=0.1), key_point("b", weight=0.2), key_point("c", weight=-0.3)
    )
    result = mark(weights, "a b c")
    assert result.breakdown is not None
    assert (result.proportion, result.breakdown.effect) == (0, "no_marks"), "0.1 + 0.2 - 0.3 is 0"
