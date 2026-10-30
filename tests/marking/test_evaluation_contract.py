from collections.abc import Iterator
from typing import Any

import pytest

from gradeflow_core import Check, MarkScheme, QuestionSet, Submission, mark
from gradeflow_core.methods.check import Context
from gradeflow_core.methods.outcome import Outcome
from gradeflow_core.questions.text import TextQuestion


def test_mark_reuses_units_exact_inputs_and_isolates_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = QuestionSet({key: TextQuestion() for key in ("Q1", "Q2", "Q3")})
    scheme = MarkScheme.load(
        [
            {"question": "Q1", "uses": "gradeflow/full-marks@1"},
            {
                "uses": "gradeflow/combination@1",
                "with": {
                    "checks": [
                        {
                            "question": "Q2",
                            "id": "gate",
                            "read_only": True,
                            "uses": "gradeflow/accepted-answers@1",
                            "with": {"answers": ["yes"]},
                        },
                        {
                            "question": "Q3",
                            "uses": "gradeflow/full-marks@1",
                            "if": "{{ checks.gate.correct }}",
                        },
                    ]
                },
            },
        ]
    )
    calls: list[str | None] = []
    original = Check.mark

    def counted(self: Check, context: Context, *args: Any, **kwargs: Any) -> dict[str, Outcome]:
        if any(self is marking for marking in scheme.markings):
            calls.append(self.question)
        return original(self, context, *args, **kwargs)

    monkeypatch.setattr(Check, "mark", counted)
    consumed: list[str] = []

    def submissions() -> Iterator[Submission]:
        for student, q1, q2 in [
            ("alice", "Same", "yes"),
            ("bob", "Same", "no"),
            ("cara", "Same", "yes"),
            ("dan", "same", "yes"),
        ]:
            consumed.append(student)
            yield Submission(student_id=student, answers={"Q1": q1, "Q2": q2, "Q3": "12"})

    stream = mark(questions, scheme, submissions())
    first = next(stream)
    assert consumed == ["alice"]
    first.results["Q1"].proportion = 0
    first.results["Q3"].feedback = "Changed by caller"
    remaining = list(stream)
    assert [result.student_id for result in remaining] == ["bob", "cara", "dan"]
    assert [result.results["Q3"].proportion for result in remaining] == [0, 1, 1]
    assert all(result.results["Q1"].proportion == 1 for result in remaining)
    assert remaining[1].results["Q3"].feedback != "Changed by caller"
    assert all(set(result.results) == {"Q1", "Q3"} for result in remaining)
    assert calls.count("Q1") == 2
    assert calls.count(None) == 2
    list(
        mark(
            questions,
            scheme,
            [Submission(student_id="again", answers={"Q1": "Same", "Q2": "yes", "Q3": "12"})],
        )
    )
    assert calls.count("Q1") == 3
    assert calls.count(None) == 3


@pytest.mark.parametrize("read_only", [True, False])
def test_single_question_marking_rejects_cross_question_dependencies(read_only: bool) -> None:
    questions = QuestionSet({key: TextQuestion() for key in ("Q1", "Q2")})
    scheme = MarkScheme.load(
        [
            {
                "question": "Q1",
                "uses": "gradeflow/combination@1",
                "with": {
                    "checks": [
                        {
                            "question": "Q2",
                            "uses": "gradeflow/full-marks@1",
                            "read_only": read_only,
                        },
                        {"uses": "gradeflow/full-marks@1"},
                    ]
                },
            }
        ]
    )
    assert any("group marking" in problem.message for problem in scheme.problems(questions))


def test_nested_group_checks_inherit_questions_and_respect_conditions_and_blanks() -> None:
    questions = QuestionSet({key: TextQuestion() for key in ("Q1", "Q2")})
    scheme = MarkScheme.load(
        [
            {
                "uses": "gradeflow/combination@1",
                "with": {
                    "checks": [
                        {
                            "question": "Q1",
                            "uses": "gradeflow/combination@1",
                            "with": {
                                "checks": [
                                    {
                                        "question": "Q2",
                                        "id": "gate",
                                        "read_only": True,
                                        "uses": "gradeflow/accepted-answers@1",
                                        "with": {"answers": ["yes"]},
                                    },
                                    {
                                        "uses": "gradeflow/full-marks@1",
                                        "if": "{{ checks.gate.correct }}",
                                    },
                                ]
                            },
                        },
                    ]
                },
            }
        ]
    )
    [group] = scheme.markings
    assert group.question is None
    assert group.questions_read() == {"Q1", "Q2"}
    assert group.questions_marked() == {"Q1"}
    assert not scheme.problems(questions)
    submissions = [
        Submission(student_id=str(i), answers={"Q1": answer, "Q2": gate})
        for i, (answer, gate) in enumerate([("x", "yes"), ("x", "no"), ("", "yes")])
    ]
    results = list(mark(questions, scheme, submissions))
    assert all(set(result.results) == {"Q1"} for result in results)
    assert [result.results["Q1"].proportion for result in results] == [1, 0, 0]
    assert results[-1].results["Q1"].feedback == "No answer."


def test_custom_code_evaluation_has_no_student_identity_and_reuses_declared_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gradeflow_core.methods.code import runner as code_runner

    calls: list[dict[str, Any]] = []

    def execute(code: str, variables: dict[str, Any], **kwargs: Any) -> None:
        calls.append(dict(variables))
        variables.update(correct=variables["answer"] == "yes")

    monkeypatch.setattr(code_runner, "run", execute)
    questions = QuestionSet({"Q1": TextQuestion()})
    scheme = MarkScheme.load(
        [
            {
                "question": "Q1",
                "uses": "gradeflow/custom-code@1",
                "with": {"code": "correct = answer == 'yes'"},
            }
        ]
    )
    submissions = [
        Submission(student_id=student, answers={"Q1": "yes"}) for student in ["alice", "bob"]
    ]
    results = list(mark(questions, scheme, submissions))
    assert [result.student_id for result in results] == ["alice", "bob"]
    assert all(result.results["Q1"].proportion == 1 for result in results)
    assert calls == [{"answer": "yes"}]
