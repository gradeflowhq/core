from typing import Any

import pytest

from gradeflow_core import BUILTINS, MarkScheme, QuestionSet
from gradeflow_core.editing import EditingError, MarkingDraft, MethodDraft
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion
from gradeflow_core.steps.step import valid_step
from tests.methods.helpers import mark_all, proportions, with_value_inputs

from ..methods.helpers import WRAPPED_NUMBER_RANGE, located

QUESTIONS = QuestionSet({"Q1": TextQuestion()})
NUMBER_QUESTIONS = QuestionSet({"Q1": NumberQuestion()})
LIMITED_RANGE = (
    "sample/limited-range",
    """label: Limited range
version: 1
question_types: [number]
settings:
  lower:
    title: Lower limit
    default: 1/4
  upper:
    title: Upper limit
    default: 3/4
steps:
  - uses: gradeflow/number-range@1
    with: {min_value: "{{ settings.lower }}", max_value: "{{ settings.upper }}"}
  - uses: gradeflow/number-range@1
    with: {min_value: "{{ settings.lower }}", max_value: 1}
""",
)
"""A method file whose settings fill limits that must stay in order."""
LIMITED_RANGE_METHODS = BUILTINS.with_files([LIMITED_RANGE])


def limited_range_draft(settings: dict[str, Any]) -> MarkingDraft:
    marking = {"question": "Q1", "uses": "sample/limited-range@1", "with": settings}
    return MarkingDraft(marking, NUMBER_QUESTIONS, "Q1", methods=LIMITED_RANGE_METHODS)


def test_any_order_values_give_their_checks_no_earlier_checks() -> None:
    questions = QuestionSet({"Q1": ValuesQuestion(value_types=["text", "text"])})
    marking = {
        "question": "Q1",
        "uses": "gradeflow/multiple-values@1",
        "with": {
            "any_order": True,
            "checks": with_value_inputs(
                {"id": "first", "uses": "gradeflow/full-marks@1"},
                {"uses": "gradeflow/full-marks@1", "if": "{{ steps.first.output.correct }}"},
            ),
        },
    }
    draft = MarkingDraft(marking, questions, "Q1")
    assert draft.earlier_checks("/with/checks/1") == []
    assert "if" not in draft.schema("/with/checks/0")["properties"]


def test_a_composed_methods_check_that_cant_mark_the_question_is_its_users_problem() -> None:
    methods = BUILTINS.with_files([WRAPPED_NUMBER_RANGE])
    draft = MarkingDraft(
        {"question": "Q1", "uses": "sample/wrapped@1"}, QUESTIONS, "Q1", methods=methods
    )
    assert located(draft.problems) == [("", "", "Number Range can't be used on a text question.")]
    assert draft.settings_holding_steps("") == []


def test_a_setting_that_breaks_a_rule_of_the_step_it_fills_is_a_problem_at_the_setting() -> None:
    draft = limited_range_draft({"upper": 0.2})
    problem = ("", "/with/upper", "The highest number is below the lowest.")
    assert located(draft.problems) == [problem]
    assert draft.describe("") == ""
    mark_scheme = MarkScheme.load([draft.data], methods=LIMITED_RANGE_METHODS)
    marking_problems = mark_scheme.problems(NUMBER_QUESTIONS)
    assert located(marking_problems) == [("/markings/0", *problem[1:])]


def test_a_rule_broken_at_a_field_the_method_file_writes_is_a_problem_of_the_step() -> None:
    draft = limited_range_draft({"lower": 1.5, "upper": 2})
    assert located(draft.problems) == [
        ("", "", "Limited range: The highest number is below the lowest.")
    ]


def test_a_cleared_setting_with_a_default_is_unset_and_takes_its_default() -> None:
    draft = limited_range_draft({"upper": 0.5})
    draft.update("", {"upper": None})
    assert draft.data["with"] == {}
    assert "Is between `0.25` and `0.75`." in draft.describe("")


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
    assert draft.problems[0].step_path == "/with/then/0"


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
    assert draft.earlier_checks("/with/then/1") == ["/with/then/0"]


def test_a_branchs_checks_read_those_before_them_while_the_condition_is_unfinished() -> None:
    questions = QuestionSet({"Q1": TextQuestion(), "Q2": TextQuestion()})
    marking = {
        "uses": "gradeflow/conditional@1",
        "with": {
            "condition": [{"question": "Q1", "uses": "gradeflow/key-points@1"}],
            "then": [
                {"id": "a", "question": "Q2", "uses": "gradeflow/full-marks@1"},
                {
                    "question": "Q2",
                    "uses": "gradeflow/full-marks@1",
                    "if": "{{ steps.a.output.correct }}",
                },
            ],
        },
    }
    draft = MarkingDraft(marking, questions, None)
    assert draft.earlier_checks("/with/then/1") == ["/with/then/0"]
    assert [problem.step_path for problem in draft.problems] == ["/with/condition/0"]


MARKING = {
    "question": "Q1",
    "uses": "gradeflow/workflow@1",
    "with": {
        "steps": [
            {"uses": "gradeflow/key-points@1", "with": {"points": ["term"]}},
            {
                "name": "No look-alike",
                "uses": "gradeflow/not-allowed@1",
                "with": {"check": {"uses": "gradeflow/key-point@1", "with": {"point": "turm"}}},
            },
        ]
    },
}


def test_a_marking_saved_as_a_method_marks_like_it_with_its_fields_as_settings() -> None:
    name = "sample/term-not-look-alike"
    draft = MethodDraft.from_marking(
        valid_step(MARKING, BUILTINS.by_ref),
        QUESTIONS,
        name=name,
        version=1,
        label="Term, not look-alike",
        summary="The term, not its look-alike.",
    )
    draft.bind_new_setting("/with/steps/0/with/points")
    draft.bind_new_setting("/with/steps/1/with/check/with/point")
    methods = BUILTINS.with_files([(name, draft.to_yaml())])
    saved = {"uses": draft.finalized().ref}
    filled = {"uses": saved["uses"], "with": {"points": ["photon"], "point": "foton"}}
    assert proportions(mark_all(saved, ["the term", "a turm term"], methods=methods)) == [1, 0]
    assert proportions(mark_all(filled, ["a photon", "a foton photon"], methods=methods)) == [1, 0]


def test_exposing_a_setting_refuses_unknown_paths() -> None:
    draft = MethodDraft.from_marking(
        valid_step(MARKING, BUILTINS.by_ref),
        QUESTIONS,
        name="x",
        version=1,
        label="X",
        summary="X.",
    )
    with pytest.raises(EditingError, match="isn't a field a setting can fill"):
        draft.bind_new_setting("/with/nope")
