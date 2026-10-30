import pytest

from gradeflow_core import Question
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from .helpers import (
    accepted,
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
    check = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": combine, "checks": [key_point("a"), key_point("b")]},
    }
    assert proportions(mark_all(check, ["a b", "a", "c"])) == expected


def test_weights_count_under_any_and_each_one_counts() -> None:
    alternatives = {
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "any",
            "checks": [
                accepted("9.81", name="Correct"),
                accepted("-9.81", name="Sign error", weight=0.5),
            ],
        },
    }
    right, sign, wrong = mark_all(alternatives, ["9.81", "-9.81", "10"], NumberQuestion())
    assert (right.proportion, sign.proportion, wrong.proportion) == (1, 0.5, 0)
    assert sign.feedback == "Sign error (50% of the marks).", "its name says why"
    assert right.feedback == "Correct.", "not Correct: Correct."
    assert wrong.feedback == "Expected 9.81.", "no check that gave nothing speaks by name"
    unnamed = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "any", "checks": [accepted("9.81"), accepted("-9.81", weight=0.5)]},
    }
    assert mark(unnamed, "-9.81", NumberQuestion()).feedback == "Partly right (50% of the marks)."
    explained = accepted(
        "-9.81", name="Sign error", weight=0.5, feedback={"correct": "Check g's direction."}
    )
    written = {"uses": "gradeflow/combination@1", "with": {"combine": "any", "checks": [explained]}}
    [sign] = mark_all(written, ["-9.81"], NumberQuestion())
    assert sign.feedback == "Check g's direction.", "as its author wrote it"
    deducting = {
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "proportional",
            "checks": [key_point("a", weight=2), key_point("x", weight=-1)],
        },
    }
    assert proportions(mark_all(deducting, ["a", "a x", "x"])) == [1, 0.5, 0]


def test_a_total_gives_its_weights_as_marks_up_to_the_questions() -> None:
    marks = {
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "total",
            "checks": [key_point(c, weight=2) for c in "abc"] + [key_point("x", weight=-1)],
        },
    }
    four = TextQuestion(max_marks=4)
    assert proportions(mark_all(marks, ["a", "a b", "a b c", "a b c x", "x"], four)) == [
        0.5,
        1,
        1,
        1,
        0,
    ]
    assert describe(marks).startswith("**Marks added up**\n\n")


def test_each_check_speaks_after_its_name_but_any_one_only_the_one_that_counted() -> None:
    every = {
        "uses": "gradeflow/combination@1",
        "with": {"checks": [key_point("a", name="A"), key_point("b")]},
    }
    assert mark(every, "a b").feedback == "A: Has “a”.\nHas “b”."
    either = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "any", "checks": [key_point("a"), key_point("b")]},
    }
    assert mark(either, "b").feedback == "Has “b”."


def test_a_penalty_speaks_only_when_it_takes_marks_off() -> None:
    titration = {
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "proportional",
            "checks": [accepted("25.10"), accepted("25.40", name="Rough titre", weight=-1)],
        },
    }
    clean, rough = mark_all(titration, ["25.10", "25.40"], NumberQuestion())
    assert clean.feedback == "Correct."
    assert rough.feedback == "Expected 25.10.\nMarks taken off: Rough titre."
    brief = {
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "proportional",
            "checks": [
                key_point("a"),
                {
                    "uses": "gradeflow/not-allowed@1",
                    "weight": -1,
                    "with": {"check": {"uses": "gradeflow/length@1", "with": {"max_length": 1}}},
                },
            ],
        },
    }
    short, long = mark_all(brief, ["a", "a b"])
    assert (short.feedback, long.feedback) == ("Has “a”.", "Has “a”.\nMarks taken off.")


def test_once_full_marks_are_reached_what_gave_nothing_stays_silent() -> None:
    three_of_four = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "total", "checks": [key_point(c) for c in "abc"]},
    }
    full, partial = mark_all(three_of_four, ["a c", "b"], TextQuestion(max_marks=2))
    assert full.feedback == "Has “a”.\nHas “c”."
    assert partial.feedback == "Missing “a”.\nHas “b”.\nMissing “c”."


def test_a_fail_is_explained_by_the_parts_that_caused_it() -> None:
    check = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "all", "checks": [key_point("a"), key_point("b")]},
    }
    assert mark(check, "a").feedback == "Missing “b”."


def test_none_may_be_correct_says_what_the_answer_did_that_is_not_allowed() -> None:
    check = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "none", "checks": [key_point("a"), key_point("b")]},
    }
    assert [result.feedback for result in mark_all(check, ["c", "a", "b a"])] == [
        "Correct.",
        "Not allowed. Includes “a”.",
        "Not allowed. Includes “a”. Includes “b”.",
    ]
    own = key_point("b", feedback={"correct": "Leave b out."})
    written = {"uses": "gradeflow/combination@1", "with": {"combine": "none", "checks": [own]}}
    assert mark(written, "b").feedback == "Not allowed. Leave b out."


def test_none_may_be_correct_ends_each_description_as_a_sentence() -> None:
    check = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "none", "checks": [accepted("a")]},
    }
    assert mark(check, "a", TextQuestion()).feedback == "Not allowed. Matches “a”."


def test_read_only_checks_decide_without_marks_and_conditions_read_earlier_results() -> None:
    marking = {
        "uses": "gradeflow/combination@1",
        "with": {
            "checks": [
                accepted("t-test", id="test", question="Q5", read_only=True),
                accepted("2.31", question="Q6", **{"if": "{{ checks.test.correct }}"}),
                accepted("5.99", question="Q6", **{"if": "{{ not checks.test.correct }}"}),
            ]
        },
    }
    questions: dict[str, Question] = {"Q5": TextQuestion(), "Q6": NumberQuestion()}
    chose_t = mark_group(marking, {"Q5": "t-test", "Q6": "2.31"}, questions)
    chose_chi = mark_group(marking, {"Q5": "chi", "Q6": "2.31"}, questions)
    assert list(chose_t) == ["Q6"]
    assert (chose_t["Q6"].proportion, chose_chi["Q6"].proportion) == (1, 0)
    assert chose_chi["Q6"].feedback == "Expected 5.99."


def test_any_one_across_questions_picks_one_check_by_its_total_marks() -> None:
    marking = {
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "any",
            "checks": [
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
            ],
        },
    }
    questions: dict[str, Question] = {"Q1": NumberQuestion(), "Q2": NumberQuestion(max_marks=2)}
    follow_through = mark_group(marking, {"Q1": "9", "Q2": "36"}, questions)
    assert {key: result.proportion for key, result in follow_through.items()} == {
        "Q1": 0.5,
        "Q2": 0.5,
    }
    mixed = mark_group(marking, {"Q1": "6", "Q2": "36"}, questions)
    assert {key: result.proportion for key, result in mixed.items()} == {"Q1": 1, "Q2": 0}


def test_it_lists_its_checks_with_their_names_and_weights() -> None:
    check = {
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "any",
            "checks": [key_point("a", name="A"), key_point("b", weight=0.5)],
        },
    }
    assert describe(check) == (
        "**Any one must be correct**\n\n"
        "- **A:** Includes `a`.\n- **50% of the marks:** Includes `b`."
    )
    counting = {
        "uses": "gradeflow/combination@1",
        "with": {
            "combine": "proportional",
            "checks": [
                key_point("a"),
                key_point("b", weight=2),
                key_point("c", name="C", weight=-1),
            ],
        },
    }
    assert describe(counting) == (
        "**Each one counts**\n\n"
        "- Includes `a`.\n"
        "- **Weight 2:** Includes `b`.\n"
        "- **C** (weight -1): Includes `c`."
    )


def test_a_combination_in_a_combination_is_listed_under_its_place() -> None:
    either = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "any", "checks": [key_point("a")]},
    }
    check = {"uses": "gradeflow/combination@1", "with": {"checks": [either, key_point("b")]}}
    assert describe({**check, "with": {"combine": "none", "checks": [key_point("a")]}}) == (
        "**None may be correct**\n\n- Includes `a`."
    )
    assert describe(check) == (
        "**All must be correct**\n\n"
        "- **Any one must be correct**\n\n  - Includes `a`.\n- Includes `b`."
    )


def test_a_combination_needs_a_check() -> None:
    assert problems_of({"uses": "gradeflow/combination@1", "with": {"checks": []}}) == [
        ("", ("checks",), "Add at least one check.")
    ]


def test_a_part_left_unmarked_says_why_even_when_it_would_take_marks_off() -> None:
    failing = {
        "uses": "gradeflow/custom-code@1",
        "with": {"code": "raise ValueError('bad')"},
        "weight": -1,
    }
    check = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "proportional", "checks": [key_point("a"), failing]},
    }
    result = mark(check, "a")
    assert not result.marked
    assert "Your code failed: ValueError: bad." in result.feedback


def test_a_part_left_unmarked_says_why_when_none_may_be_correct() -> None:
    failing = {"uses": "gradeflow/custom-code@1", "with": {"code": "raise ValueError('bad')"}}
    check = {
        "uses": "gradeflow/combination@1",
        "with": {"combine": "none", "checks": [key_point("a"), failing]},
    }
    result = mark(check, "c")
    assert not result.marked
    assert "Your code failed: ValueError: bad." in result.feedback
    assert "Correct." not in result.feedback
