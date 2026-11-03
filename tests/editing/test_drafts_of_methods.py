from typing import Any

import pytest

from gradeflow_core import BUILTINS, Check, QuestionSet
from gradeflow_core.editing import EditingError, MarkingDraft, MethodDraft
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion
from tests.methods.helpers import mark_all, proportions

QUESTIONS = QuestionSet({"Q1": TextQuestion()})


def test_any_order_values_give_their_checks_no_earlier_checks() -> None:
    questions = QuestionSet({"Q1": ValuesQuestion(value_types=["text", "text"])})
    marking = {
        "question": "Q1",
        "uses": "gradeflow/multiple-values@1",
        "with": {
            "any_order": True,
            "checks": [
                {"id": "first", "uses": "gradeflow/full-marks@1"},
                {"uses": "gradeflow/full-marks@1", "if": "{{ checks.first.correct }}"},
            ],
        },
    }
    draft = MarkingDraft(marking, questions, "Q1")
    assert draft.earlier_checks("with.checks.1") == []
    assert "if" not in draft.schema("with.checks.0")["properties"]


def test_a_composed_methods_check_that_cant_mark_the_question_is_its_users_problem() -> None:
    methods = BUILTINS.with_files(
        [
            (
                "sample/wrapped",
                """label: Wrapped
version: 1
summary: A wrapped number range.
question_types: [text]
checks:
  - uses: gradeflow/number-range@1
    with: {min_value: 1}
""",
            )
        ]
    )
    draft = MarkingDraft(
        {"question": "Q1", "uses": "sample/wrapped@1"}, QUESTIONS, "Q1", methods=methods
    )
    assert [(problem.check, problem.field, problem.message) for problem in draft.problems] == [
        ("", (), "Number Range can't mark a text question.")
    ]
    assert draft.checks_settings("") == []


def test_a_drafts_problems_are_where_the_check_is_in_a_branch() -> None:
    questions = QuestionSet({"Q1": TextQuestion(), "Q2": TextQuestion()})
    marking: dict[str, Any] = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [{"question": "Q1", "uses": "gradeflow/full-marks@1"}],
            "then": [
                {"question": "Q2", "uses": "gradeflow/number-range@1", "with": {"min_value": 1}}
            ],
        },
    }
    draft = MarkingDraft(marking, questions, None)
    assert draft.problems[0].check == "with.then.0"


def test_a_drafts_earlier_checks_match_generated_execution_order() -> None:
    marking = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [{"question": "Q1", "uses": "gradeflow/full-marks@1"}],
            "then": [
                {"question": "Q1", "uses": "gradeflow/full-marks@1"},
                {"question": "Q1", "uses": "gradeflow/full-marks@1"},
            ],
        },
    }
    draft = MarkingDraft(marking, QUESTIONS, None)
    assert not draft.problems
    assert draft.earlier_checks("with.then.1") == ["with.then.0"]


MARKING = {
    "question": "Q1",
    "uses": "gradeflow/combination@1",
    "with": {
        "checks": [
            {"uses": "gradeflow/key-points@1", "with": {"points": ["term"]}},
            {
                "name": "No look-alike",
                "uses": "gradeflow/not-allowed@1",
                "with": {"check": {"uses": "gradeflow/key-point@1", "with": {"point": "turm"}}},
            },
        ]
    },
}


def test_a_marking_saved_as_a_method_marks_like_it_with_its_fields_as_inputs() -> None:
    name = "sample/term-not-look-alike"
    draft = MethodDraft.from_marking(
        Check.load(MARKING),
        QUESTIONS,
        name=name,
        version=1,
        label="Term, not look-alike",
        summary="The term, not its look-alike.",
    )
    draft.bind_new_input("with.checks.0.with.points", label="Points", default=["term"])
    draft.bind_new_input("with.checks.1.with.check.with.point", label="Look-alike", default="turm")
    methods = BUILTINS.with_files([(name, draft.to_yaml())])
    saved = {"uses": draft.ref}
    filled = {"uses": draft.ref, "with": {"points": ["photon"], "point": "foton"}}
    assert proportions(mark_all(saved, ["the term", "a turm term"], methods=methods)) == [1, 0]
    assert proportions(mark_all(filled, ["a photon", "a foton photon"], methods=methods)) == [1, 0]


def test_exposing_a_setting_refuses_unknown_paths() -> None:
    draft = MethodDraft.from_marking(
        Check.load(MARKING), QUESTIONS, name="x", version=1, label="X", summary="X."
    )
    with pytest.raises(EditingError, match="isn't a setting an input can fill"):
        draft.bind_new_input("with.nope", label="Missing")
