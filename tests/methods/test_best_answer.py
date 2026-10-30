from typing import Any

from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.reading import CodeReading

from .helpers import CODE, code_tests, describe, mark, mark_all, problems_of

GUESSING = CodeQuestion(reading=CodeReading(guess_indentation=True))
LONGEST_RUN = (
    "def longest_run(xs):\nbest = 0\nrun = 0\nprev = 0\nfor x in xs:\nif x == prev:\nrun += 1\n"
    "else:\nrun = 1\nbest = max(best, run)\nprev = x\nreturn best"
)
"""Flat code whose likeliest layout puts best = max(best, run) after the loop, wrongly."""


def best(check: dict[str, Any]) -> dict[str, Any]:
    return {"uses": "gradeflow/best-answer@1", "with": {"check": check}}


RUN = code_tests({"expression": "longest_run([1, 1, 1, 2])", "expected": "3"})


def test_the_layout_with_the_most_marks_counts_and_feedback_shows_it() -> None:
    result = mark(best(RUN), LONGEST_RUN, GUESSING)
    assert result.proportion == 1
    assert result.method == "Code Tests"
    assert result.feedback.startswith("Read as:\ndef longest_run(xs):\n")
    assert "\n        best = max(best, run)\n" in result.feedback
    assert result.feedback.endswith("\n\nlongest_run([1, 1, 1, 2]) gave 3, as expected.")


def test_on_a_tie_the_likeliest_layout_counts_without_a_word() -> None:
    [right, other] = mark_all(
        best(code_tests({"expression": "len(longest_run.__name__)", "expected": "11"})),
        [LONGEST_RUN, "def longest_run(xs):\nreturn 0"],
        GUESSING,
    )
    assert (right.proportion, right.feedback) == (
        1,
        "len(longest_run.__name__) gave 11, as expected.",
    )
    assert other.proportion == 1


def test_a_layout_the_check_marks_counts_over_one_it_cant() -> None:
    custom = {
        "uses": "gradeflow/custom-code@1",
        "with": {"code": "if '\\n    best = max' in answer:\n    raise ValueError\ncorrect = True"},
    }
    result = mark(best(custom), LONGEST_RUN, GUESSING)
    assert (result.marked, result.correct) == (True, True)


def test_a_check_that_marks_one_answer_fits_inside_best_answer_only() -> None:
    assert problems_of(RUN, GUESSING) == [
        (
            "",
            (),
            "Code Tests marks one answer, and this question reads each answer as several. "
            "Put it inside Best Answer.",
        )
    ]
    assert problems_of(best(RUN), GUESSING) == []


def test_best_answer_needs_a_question_read_as_several() -> None:
    assert problems_of(best(RUN), CODE) == [
        ("", (), "Best Answer marks an answer read as several, which this question doesn't.")
    ]


def test_it_describes_the_check_inside() -> None:
    assert describe(best(RUN)) == (
        "**Best possible answer:** Gives `3` for `longest_run([1, 1, 1, 2])`."
    )
