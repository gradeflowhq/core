from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import Field

from gradeflow_core import (
    BUILTINS,
    Answer,
    AnswerMethod,
    Detail,
    InvalidError,
    MarkScheme,
    Methods,
    Outcome,
    Question,
    QuestionSet,
    Settings,
    Span,
    Submission,
    Transformed,
    TransformMethod,
    Unreadable,
    mark,
)
from gradeflow_core.editing import Position, compatible_methods

QUESTIONS = QuestionSet.load({"Q1": {"type": "text", "max_marks": 2}})


class StartsWithSettings(Settings):
    prefix: str = Field(title="Starts with")


class StartsWith(AnswerMethod[StartsWithSettings]):
    name = "sample/starts-with"
    version = 1
    label = "Starts With"
    summary = "Correct if the answer starts with the text."
    question_types = frozenset({"text"})
    settings = StartsWithSettings

    def describe(self, settings: StartsWithSettings) -> str:
        return f"Starts with `{settings.prefix}`."

    def mark_answer(
        self, settings: StartsWithSettings, answer: Answer, question: Question
    ) -> Outcome:
        if str(answer).startswith(settings.prefix):
            return Outcome(1.0, "Correct.")
        if not str(answer).isascii():
            return Outcome(0.0, "Can't read it.", marked=False)
        return Outcome(0.0, "Wrong start.", details=(Detail("Starts with", str(answer)[:1]),))


class AfterSettings(Settings):
    marker: str = Field(title="After")


class After(TransformMethod[AfterSettings]):
    name = "sample/after"
    version = 1
    label = "After"
    summary = "Finds what comes after the text."
    question_types = frozenset({"text"})
    settings = AfterSettings

    def describe(self, settings: AfterSettings) -> str:
        return f"Finds what comes after `{settings.marker}`."

    def output_question(self, settings: AfterSettings, question: Question) -> Question:
        return question

    def transform(
        self, settings: AfterSettings, answer: Answer, question: Question
    ) -> Transformed | Unreadable:
        _, marker, rest = str(answer).partition(settings.marker)
        return (
            Transformed(rest.strip()) if marker else Unreadable(f"Couldn't find {settings.marker}.")
        )


class SpanPastTheEnd(After):
    name = "sample/span-past-the-end"

    def transform(self, settings: AfterSettings, answer: Answer, question: Question) -> Transformed:
        return Transformed(answer, (Span(0, len(str(answer)) + 1),))


METHODS = Methods([*BUILTINS, StartsWith(), After(), SpanPastTheEnd()])
STARTS_WITH = {"question": "Q1", "uses": "sample/starts-with@1", "with": {"prefix": "photo"}}


def results_of(marking: Mapping[str, Any], answers: list[str]) -> list[tuple[float, str, bool]]:
    scheme = MarkScheme.load([marking], methods=METHODS)
    assert scheme.problems(QUESTIONS) == []
    submissions = [Submission(student_id=answer, answers={"Q1": answer}) for answer in answers]
    return [
        (result.marks, result.feedback, result.is_marked)
        for marked in mark(QUESTIONS, scheme, submissions)
        for result in marked.results.values()
    ]


def test_a_method_written_in_python_marks_describes_and_is_offered() -> None:
    assert results_of(STARTS_WITH, ["photosynthesis", "respiration", "", "φῶς"]) == [
        (2, "Correct.", True),
        (0, "Wrong start.", True),
        (0, "No answer.", True),
        (0, "Can't read it.", False),
    ]
    [check] = MarkScheme.load([STARTS_WITH], methods=METHODS).markings
    assert check.describe() == "Starts with `photo`."
    method = METHODS.find("sample/starts-with@1")
    assert method in compatible_methods(Position(QUESTIONS, "Q1"), methods=METHODS)
    assert method.gives_marks({})
    assert list(method.settings_schema(Position(QUESTIONS, "Q1"))["properties"]) == ["prefix"]


@pytest.mark.parametrize("outcome", [Outcome(float("nan"), "Share."), Outcome(1.0, None), 1.0])  # type: ignore[arg-type]
def test_an_outcome_that_isnt_valid_leaves_only_its_answer_for_a_person(outcome: Any) -> None:
    class Broken(StartsWith):
        name = "sample/broken"

        def mark_answer(
            self, settings: StartsWithSettings, answer: Answer, question: Question
        ) -> Outcome:
            return outcome if answer == "bad" else Outcome(1.0, "Correct.")

    scheme = MarkScheme.load(
        [{**STARTS_WITH, "uses": "sample/broken@1"}], methods=Methods([*BUILTINS, Broken()])
    )
    submissions = [Submission(student_id=answer, answers={"Q1": answer}) for answer in ("bad", "a")]
    assert [
        (marked.results["Q1"].feedback, marked.results["Q1"].is_marked)
        for marked in mark(QUESTIONS, scheme, submissions)
    ] == [("Couldn't mark this answer.", False), ("Correct.", True)]


def test_its_settings_are_validated_as_a_built_in_methods_are() -> None:
    with pytest.raises(InvalidError, match=r"/markings/0/with/prefix: Required\."):
        MarkScheme.load([{**STARTS_WITH, "with": {}}], methods=METHODS)


def test_a_transform_written_in_python_gives_later_steps_their_input() -> None:
    workflow = {
        "question": "Q1",
        "uses": "gradeflow/workflow@1",
        "with": {
            "steps": [
                {"id": "final", "uses": "sample/after@1", "with": {"marker": "answer:"}},
                {
                    "input": "{{ steps.final.output }}",
                    "uses": "sample/starts-with@1",
                    "with": {"prefix": "photo"},
                },
            ]
        },
    }
    assert not METHODS.find("sample/after@1").gives_marks({})
    assert results_of(workflow, ["I think. Answer: photosynthesis", "photosynthesis"]) == [
        (2, "Correct.", True),
        (0, "Couldn't find answer:.", False),
    ]


def test_a_transform_giving_a_span_past_its_input_leaves_the_answer_for_a_person() -> None:
    workflow = {
        "question": "Q1",
        "uses": "gradeflow/workflow@1",
        "with": {
            "steps": [
                {"id": "after", "uses": "sample/span-past-the-end@1", "with": {"marker": ""}},
                {**STARTS_WITH, "input": "{{ steps.after.output }}"},
            ]
        },
    }
    assert results_of(workflow, ["photosynthesis"]) == [(0, "Couldn't mark this answer.", False)]


class CheckWithSpans(StartsWith):
    name = "sample/check-with-spans"

    def mark_answer(
        self, settings: StartsWithSettings, answer: Answer, question: Question
    ) -> Outcome:
        return Outcome(1.0, "Correct.", spans=(Span(0, len(settings.prefix)),))


def test_spans_a_check_gives_are_dropped() -> None:
    scheme = MarkScheme.load(
        [{**STARTS_WITH, "uses": "sample/check-with-spans@1"}],
        methods=Methods([*BUILTINS, CheckWithSpans()]),
    )
    [marked] = mark(QUESTIONS, scheme, [Submission(student_id="1", answers={"Q1": "photo"})])
    assert marked.results["Q1"].spans == ()
