import pytest

from gradeflow_core.questions.question_set import QuestionSet
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.steps.combining import COMBINATIONS, MarksBy, Part, combine
from gradeflow_core.steps.outcome import Detail, Effect, Outcome, QuestionOutcomes, Skipped

QUESTIONS = QuestionSet({"Q1": TextQuestion(max_marks=4), "Q2": TextQuestion(max_marks=2)})
CORRECT, INCORRECT = Outcome(1.0, "Correct."), Outcome(0.0, "Expected x.")
UNMARKED = Outcome(0.0, "Couldn't decide.", marked=False)


def combined(marks_by: MarksBy, *checks: tuple[Part, Outcome | Skipped]) -> Outcome:
    """Q1's outcome from each check's result for it."""
    return combined_group(marks_by, *((part, {"Q1": result}) for part, result in checks))["Q1"]


def combined_group(marks_by: MarksBy, *checks: tuple[Part, QuestionOutcomes]) -> dict[str, Outcome]:
    outcomes = combine(marks_by, checks, QUESTIONS)
    assert all(isinstance(outcome, Outcome) for outcome in outcomes.values())
    return {
        question_id: outcome
        for question_id, outcome in outcomes.items()
        if isinstance(outcome, Outcome)
    }


def effects(outcome: Outcome) -> list[tuple[Effect, float | None]]:
    return [(child.effect, child.contribution) for child in outcome.children]


@pytest.mark.parametrize(
    ("marks_by", "proportion", "feedback"),
    [
        ("all_correct", 0.0, "Second: Expected x."),
        ("any_correct", 1.0, "Correct."),
        ("none_correct", 0.0, "Not allowed. Is “x”."),
        ("total", 0.25, "First: Correct.\nSecond: Expected x."),
        ("proportional", 0.5, "First: Correct.\nSecond: Expected x."),
    ],
)
def test_each_way_of_giving_marks_has_its_proportion_and_feedback(
    marks_by: MarksBy, proportion: float, feedback: str
) -> None:
    outcome = combined(
        marks_by,
        (Part(1, name="First", describe=lambda: "Is `x`"), CORRECT),
        (Part(1, name="Second"), INCORRECT),
    )
    assert (outcome.proportion, outcome.feedback) == (proportion, feedback)
    assert [part.name for part in outcome.parts] == ["First", "Second"]


@pytest.mark.parametrize(
    ("marks_by", "expected"),
    [
        ("all_correct", [("earned", None), ("no_marks", None), ("no_marks", None)]),
        ("any_correct", [("earned", None), ("uncounted", None), ("uncounted", None)]),
        ("none_correct", [("penalty", None), ("uncounted", None), ("uncounted", None)]),
        ("total", [("earned", 0.25), ("no_marks", 0.0), ("uncounted", 0.0)]),
        ("proportional", [("earned", 0.5), ("no_marks", 0.0), ("uncounted", 0.0)]),
    ],
)
def test_each_part_says_what_it_did_to_the_marks(
    marks_by: MarksBy, expected: list[tuple[Effect, float | None]]
) -> None:
    outcome = combined(marks_by, (Part(1), CORRECT), (Part(1), INCORRECT), (Part(-1), INCORRECT))
    assert effects(outcome) == expected


def test_a_penalty_says_only_that_it_applies_unless_its_feedback_is_its_own() -> None:
    def feedback(penalty: Part) -> str:
        return combined("total", (Part(1), CORRECT), (penalty, CORRECT)).feedback

    assert feedback(Part(-1, name="Sign error")) == "Penalty: Sign error."
    assert feedback(Part(-1, default_name="Value 2")) == "Penalty."
    assert feedback(Part(-1, name="Sign error", has_own_feedback=True)) == ("Sign error: Correct.")


def test_a_check_of_no_weight_says_whether_its_condition_is_met_and_gives_no_marks() -> None:
    outcome = combined("all_correct", (Part(0), INCORRECT), (Part(1), CORRECT))
    assert outcome.correct
    assert effects(outcome) == [("not_met", None), ("earned", None)]
    assert [part.outcome for part in outcome.parts] == [CORRECT]


def test_a_skipped_check_isnt_combined_and_a_question_with_only_those_is_skipped() -> None:
    outcome = combined("all_correct", (Part(1), Skipped()), (Part(1), CORRECT))
    assert outcome.correct
    assert [child.outcome for child in outcome.children] == [CORRECT]
    only_skipped: list[tuple[Part, QuestionOutcomes]] = [
        (Part(1), {"Q1": Skipped()}),
        (Part(0), {"Q1": CORRECT}),
    ]
    assert combine("all_correct", only_skipped, QUESTIONS) == {"Q1": Skipped()}


def test_any_one_counts_the_check_giving_the_most_marks_over_its_questions() -> None:
    outcomes = combined_group(
        "any_correct",
        (Part(1, name="Both"), {"Q1": INCORRECT, "Q2": CORRECT}),
        (Part(0.75, name="Carried"), {"Q1": CORRECT, "Q2": CORRECT}),
        (Part(1, name="Only first"), {"Q1": CORRECT}),
    )
    assert {question_id: outcome.proportion for question_id, outcome in outcomes.items()} == {
        "Q1": 0.75,
        "Q2": 0.75,
    }
    assert outcomes["Q1"].feedback == "Carried (75% of the marks)."
    assert effects(outcomes["Q1"]) == [("uncounted", None), ("earned", None), ("uncounted", None)]


def test_an_answer_stays_unmarked_while_a_check_that_couldnt_mark_it_could_give_more() -> None:
    outcome = combined(
        "any_correct", (Part(1, name="Full"), UNMARKED), (Part(0.5, name="Half"), CORRECT)
    )
    assert (outcome.proportion, outcome.marked) == (0.5, False)
    assert outcome.feedback == "Full: Couldn't decide.\nHalf (50% of the marks)."


def test_none_may_be_correct_gives_no_marks_when_a_part_couldnt_be_checked() -> None:
    failed = Outcome(0.0, "Failed.", details=(Detail("Output", "ZeroDivisionError", "error"),))
    outcome = combined("none_correct", (Part(1), failed))
    assert (outcome.proportion, outcome.feedback) == (
        0.0,
        "Couldn't check the answer: ZeroDivisionError.",
    )


@pytest.mark.parametrize(
    ("marks_by", "notes", "problem_count"),
    [
        ("all_correct", (), 1),
        ("none_correct", (), 1),
        ("any_correct", ("50% of the marks",), 0),
        ("total", ("weight 0.5",), 0),
        ("proportional", ("weight 0.5",), 0),
    ],
)
def test_weights_are_noted_where_they_count_and_problems_where_they_dont(
    marks_by: MarksBy, notes: tuple[str, ...], problem_count: int
) -> None:
    combination = COMBINATIONS[marks_by]
    assert combination.weight_notes(0.5) == notes
    assert combination.weight_notes(1) == ()
    problems = list(combination.weight_problems([1, 0, 0.5], "steps"))
    assert [problem.field for problem in problems] == [("steps", 2, "weight")] * problem_count
