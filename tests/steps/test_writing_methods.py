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
    Submission,
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
    summary = "Full marks if the answer starts with the text."
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
    ) -> Answer | Unreadable:
        _, marker, rest = str(answer).partition(settings.marker)
        return rest.strip() if marker else Unreadable(f"Couldn't find {settings.marker}.")


METHODS = Methods([*BUILTINS, StartsWith(), After()])
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
