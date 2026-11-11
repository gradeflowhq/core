from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import Mock

import pytest

from gradeflow_core import (
    BUILTINS,
    GradeFlowError,
    MarkScheme,
    Methods,
    QuestionResult,
    QuestionSet,
    ResultsCsv,
    Submission,
    mark,
)
from gradeflow_core.methods.code import runner as code_runner
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.types import QuestionId
from gradeflow_core.questions.values import ValuesQuestion
from gradeflow_core.steps.context import Context
from gradeflow_core.steps.method import MarkingMethod
from gradeflow_core.steps.outcome import Outcome, QuestionOutcomes
from gradeflow_core.steps.settings import Settings
from tests.methods.helpers import accepted_answer, mark_all, with_value_inputs

QUESTIONS = QuestionSet(
    {"Q1": TextQuestion(), "Q2": NumberQuestion(max_marks=2), "Q3": TextQuestion()}
)


def results_of(*markings: Mapping[str, Any], **answers: str) -> dict[str, QuestionResult]:
    scheme = MarkScheme.load(list(markings))
    [submission] = mark(QUESTIONS, scheme, [Submission(student_id="s1", answers=answers)])
    return submission.results


def test_each_question_is_marked_by_its_marking_and_others_get_no_result() -> None:
    results = results_of(
        accepted_answer("a", question="Q1"),
        accepted_answer("4", question="Q2"),
        accepted_answer("x", question="Q9"),
        Q1="a",
        Q2="5",
        Q3="c",
    )
    assert {
        question_id: (result.proportion, result.feedback, result.method_label)
        for question_id, result in results.items()
    } == {
        "Q1": (1, "Correct.", "Accepted Answers"),
        "Q2": (0, "Expected 4.", "Accepted Answers"),
    }


def test_a_question_marked_twice_gets_its_first_marking() -> None:
    results = results_of(
        accepted_answer("a", question="Q1"), accepted_answer("b", question="Q1"), Q1="a"
    )
    assert results["Q1"].proportion == 1


@pytest.mark.parametrize("answer", ["", "N/A", "  "])
def test_a_blank_answer_gets_no_marks(answer: str) -> None:
    result = results_of(accepted_answer("a", question="Q1"), Q1=answer)["Q1"]
    assert (result.proportion, result.feedback, result.is_marked) == (0, "No answer.", True)


def test_a_missing_answer_is_blank() -> None:
    assert results_of(accepted_answer("a", question="Q1"))["Q1"].feedback == "No answer."


def test_an_answer_the_question_cant_read_is_left_for_a_person() -> None:
    result = results_of(accepted_answer("4", question="Q2"), Q2="approx 4")["Q2"]
    assert (result.proportion, result.is_marked) == (0, False)
    assert result.feedback == "Couldn't read “approx 4” as a number."


def test_an_answer_with_more_values_than_the_question_has_is_left_for_a_person() -> None:
    questions = QuestionSet({"Q1": ValuesQuestion(value_types=["number", "number"])})
    values = {
        "question": "Q1",
        "uses": "gradeflow/multiple-values@1",
        "with": {"checks": with_value_inputs(accepted_answer("1"), accepted_answer("2"))},
    }
    [result] = mark_all(values, ["1, 2, 3"], questions["Q1"])
    assert (result.is_marked, result.feedback) == (False, "Expected 2 values, but got 3.")


def test_what_a_condition_on_an_answer_it_cant_read_decides_is_left_for_a_person() -> None:
    marking = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [accepted_answer("1", question="Q2")],
            "then": [accepted_answer("a", question="Q1")],
            "otherwise": [accepted_answer("b", question="Q1")],
        },
    }
    result = results_of(marking, Q1="b", Q2="approx 1")["Q1"]
    assert (result.proportion, result.is_marked) == (0, False)
    assert result.feedback == "Couldn't read “approx 1” as a number."


@pytest.mark.parametrize(
    ("error", "feedback"),
    [
        (GradeFlowError("No sandbox."), "Couldn't mark this answer: No sandbox."),
        (KeyError("Q3"), "Couldn't mark this answer."),
    ],
)
def test_a_marking_that_fails_leaves_its_questions_for_a_person_saying_why_only_if_known(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    feedback: str,
) -> None:
    monkeypatch.setattr(code_runner, "run", Mock(side_effect=error))
    custom = {"question": "Q1", "uses": "gradeflow/custom-code@1", "with": {"code": "x = 1"}}
    result = results_of(custom, Q1="a")["Q1"]
    assert (result.proportion, result.is_marked, result.feedback) == (0, False, feedback)
    assert bool(caplog.records) == (not isinstance(error, GradeFlowError)), "a bug is logged"


def test_a_marking_with_problems_leaves_its_questions_for_a_person_saying_why() -> None:
    results = results_of(
        accepted_answer("four", question="Q2"), accepted_answer("a", question="Q1"), Q1="a", Q2="4"
    )
    assert (results["Q2"].proportion, results["Q2"].is_marked) == (0, False)
    assert results["Q2"].feedback == "answers: Couldn't read “four” as a number."
    assert results["Q1"].is_marked


def test_an_empty_workflow_leaves_its_question_for_a_person_saying_why() -> None:
    empty = {"question": "Q1", "uses": "gradeflow/workflow@1", "with": {"steps": []}}
    result = results_of(empty, Q1="a")["Q1"]
    assert (result.proportion, result.is_marked) == (0, False)
    assert result.feedback == "steps: Add at least one step."


def test_results_hold_a_proportion_of_the_questions_marks() -> None:
    result = QuestionResult(proportion=0.1 + 0.2, max_marks=3, method_label="Key Points")
    assert result.marks == pytest.approx(0.9)
    assert QuestionResult(proportion=1 - 1e-12, max_marks=1).correct
    assert not QuestionResult(proportion=0.99, max_marks=1).correct


def test_each_result_and_the_total_count_the_questions_maximum_marks() -> None:
    scheme = MarkScheme.load(
        [accepted_answer("a", question="Q1"), accepted_answer("4", question="Q2")]
    )
    [submission] = mark(QUESTIONS, scheme, [Submission(student_id="s1", answers={"Q2": "4"})])
    assert (submission.results["Q2"].marks, submission.results["Q2"].max_marks) == (2, 2)
    # Q3 has no marking yet, and still counts.
    assert (submission.total_marks, submission.max_marks) == (2, 4)


def test_a_question_without_a_marking_keeps_its_given_marks() -> None:
    scheme = MarkScheme.load([accepted_answer("a", question="Q1")])
    with_given_marks = Submission(
        student_id="s1", answers={"Q1": "b"}, marks={"Q1": 1, "Q2": 1.5, "Q9": 1}
    )
    [submission] = mark(QUESTIONS, scheme, [with_given_marks])
    assert submission.results["Q1"].marks == 0, "a marking's result replaces given marks"
    assert submission.results["Q2"] == QuestionResult(proportion=0.75, max_marks=2)
    assert "Q9" not in submission.results
    over_max_marks = Submission(student_id="s2", answers={}, marks={"Q2": 5})
    assert next(mark(QUESTIONS, scheme, [over_max_marks])).results["Q2"].marks == 2


def test_a_marked_submission_is_the_submission_with_its_results() -> None:
    with_given_marks = Submission(student_id="s1", answers={"Q1": "a"}, marks={"Q2": 1})
    [submission] = mark(
        QUESTIONS, MarkScheme.load([accepted_answer("a", question="Q1")]), [with_given_marks]
    )
    assert submission.sheet is with_given_marks
    assert set(submission.results) == {"Q1", "Q2"}
    row = ResultsCsv().write(QUESTIONS, [submission]).splitlines()[1]
    assert row.startswith("s1,a,1,Correct.")


class BatchRecordingMethod(MarkingMethod[Settings]):
    """Says which version marked each answer, and records what it was asked to mark together."""

    label = "Recording Batches"
    summary = "Its version for each answer."
    question_types = frozenset({"text"})
    settings = Settings
    needs_model_endpoint = True

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def describe(self, settings: Settings) -> str:
        return self.summary

    def mark_students(
        self, settings: Settings, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> list[QuestionOutcomes]:
        assert question_id is not None
        self.batches.append([context.written_answers[question_id] for context in contexts])
        return super().mark_students(settings, contexts, question_id)

    def mark_student(
        self, settings: Settings, context: Context, question_id: QuestionId | None
    ) -> dict[QuestionId, Outcome]:
        assert question_id is not None
        return {question_id: Outcome(1.0, f"Saw {self.version}.")}


class BatchRecordingMethodVersion1(BatchRecordingMethod):
    name = "test/recording-batches"
    version = 1


class BatchRecordingMethodVersion2(BatchRecordingMethod):
    name = "test/recording-batches"
    version = 2


def test_each_version_of_a_method_marks_its_own_steps() -> None:
    first, second = BatchRecordingMethodVersion1(), BatchRecordingMethodVersion2()
    steps_of_both_versions = [{"uses": f"test/recording-batches@{version}"} for version in (1, 2)]
    marking = {
        "question": "Q1",
        "uses": "gradeflow/workflow@1",
        "with": {"steps": steps_of_both_versions},
    }
    [result] = mark_all(marking, ["x"], methods=Methods([*BUILTINS, first, second]))
    assert result.feedback == "Saw 1.\nSaw 2."
    assert (first.batches, second.batches) == ([["x"]], [["x"]])


def test_a_method_is_given_each_distinct_answer_once_in_one_batch() -> None:
    method = BatchRecordingMethodVersion1()
    marking = {"question": "Q1", "uses": "test/recording-batches@1"}
    results = mark_all(marking, ["a", "b", "a"], methods=Methods([*BUILTINS, method]))
    assert method.batches == [["a", "b"]]
    assert [result.feedback for result in results] == ["Saw 1."] * 3


def test_a_marking_of_a_question_not_supplied_marks_nothing() -> None:
    method = BatchRecordingMethodVersion1()
    scheme = MarkScheme.load(
        [{"question": "Q9", "uses": "test/recording-batches@1"}], methods=Methods([method])
    )
    [submission] = mark(QUESTIONS, scheme, [Submission(student_id="s1", answers={"Q9": "x"})])
    assert "Q9" not in submission.results
    assert method.batches == []
