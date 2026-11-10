"""Declared setting defaults are authored once, independently of test values."""

import re

import pytest

from gradeflow_core import BUILTINS, InvalidError, Methods, Problem, QuestionSet
from gradeflow_core.editing import EditingError, ExampleRow, MethodDraft, load_yaml, mark_rows

from .helpers import NUMBER_QUESTIONS, number_range_draft


def test_binding_a_new_setting_inherits_current_value_without_creating_a_test_value() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("with.steps.0.with.min_value")
    assert draft.declared_settings["min_value"]["default"] == 10
    assert draft.test_values == {}
    draft.update_declared_setting("min_value", {"title": "Minimum", "default": 12})
    assert draft.steps_draft.values("with.steps.0")["min_value"] == 12
    draft.steps_draft.update("with.steps.0", {"min_value": 14})
    assert draft.declared_settings["min_value"]["default"] == 14
    assert load_yaml(draft.to_yaml())["steps"][0]["with"]["min_value"] == "{{ settings.min_value }}"


def test_a_condition_may_read_the_methods_settings() -> None:
    draft = number_range_draft()
    assert "if" not in draft.steps_draft.schema("with.steps.0")["properties"]
    draft.bind_new_setting("with.steps.0.with.min_value")
    assert "if" in draft.steps_draft.schema("with.steps.0")["properties"]
    draft.steps_draft.update("with.steps.0", {"if": "{{ settings.min_value > 0 }}"})
    assert draft.steps_draft.problems == []
    assert draft.problems == []


def test_test_values_never_replace_authored_defaults_or_fixed_values() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("with.steps.0.with.min_value")
    draft.test_values = {"min_value": 99}
    assert draft.steps_draft.values("with.steps.0")["min_value"] == 10
    draft.steps_draft.update("with.steps.0", {"min_value": 12})
    assert draft.test_values == {"min_value": 99}
    draft.unbind("with.steps.0.with.min_value")
    assert draft.steps_draft.values("with.steps.0")["min_value"] == 12
    assert draft.test_values == {}


def test_a_setting_with_no_default_is_checked_with_its_test_value() -> None:
    questions = QuestionSet.load({"Q1": {"type": "text"}})
    draft = MethodDraft.new(questions, name="answers", version=1, label="Answers")
    draft.steps_draft.add("with.steps.0", "gradeflow/accepted-answers@1")
    draft.steps_draft.add("with.steps.1", "gradeflow/equivalent-expression@1")
    draft.steps_draft.update("with.steps.1", {"answers": ["x2"]})
    draft.bind_new_setting("with.steps.0.with.answers")
    assert draft.steps_draft.problems == []
    assert [str(problem) for problem in draft.problems] == [
        "Give “Accepted answers” a test value or a default."
    ]
    draft.test_values = {"answers": ["sun"]}
    expected = [Problem("Unknown name “x2”.", "with.steps.1", ("answers", 0))]
    assert draft.steps_draft.problems == expected
    assert draft.problems == expected
    assert draft.steps_draft.values("with.steps.0")["answers"] is None


@pytest.mark.parametrize("value", [None, "", [], {}])
def test_empty_values_create_required_settings(value: object) -> None:
    draft = number_range_draft()
    marking = draft.steps_draft.data
    marking["with"]["steps"][0]["with"]["min_value"] = value
    draft.steps_draft.replace(marking)
    draft.bind_new_setting("with.steps.0.with.min_value")
    assert "default" not in draft.declared_settings["min_value"]
    assert draft.steps_draft.values("with.steps.0")["min_value"] is None


def test_zero_and_false_are_defaults_and_clearing_a_default_is_required() -> None:
    draft = number_range_draft()
    draft.steps_draft.update("with.steps.0", {"min_value": 0})
    draft.bind_new_setting("with.steps.0.with.min_value")
    assert draft.declared_settings["min_value"]["default"] == 0
    draft.steps_draft.update("with.steps.0", {"min_value": None})
    assert "default" not in draft.declared_settings["min_value"]
    text_draft = MethodDraft.new(
        QuestionSet.load({"Q1": {"type": "text"}}), name="text", version=1, label="Text"
    )
    text_draft.steps_draft.add("with.steps.0", "gradeflow/key-points@1")
    text_draft.bind_new_setting("with.steps.0.with.whole_words")
    assert text_draft.declared_settings["whole_words"]["default"] is False


def test_binding_existing_setting_preserves_default_and_updates_every_bound_field() -> None:
    draft = number_range_draft()
    min_value_path = "with.steps.0.with.min_value"
    max_value_path = "with.steps.0.with.max_value"
    draft.bind_new_setting(min_value_path)
    draft.bind_new_setting(max_value_path)
    assert "min_value" in draft.compatible_settings(max_value_path)
    draft.bind(max_value_path, "min_value")
    assert set(draft.declared_settings) == {"min_value"}
    assert draft.steps_draft.values("with.steps.0")["max_value"] == 10
    # An update with both fields; the unchanged sibling must not undo the edit.
    draft.steps_draft.update("with.steps.0", {"min_value": 12, "max_value": 10})
    assert draft.steps_draft.values("with.steps.0")["min_value"] == 12
    assert draft.steps_draft.values("with.steps.0")["max_value"] == 12
    draft.unbind(min_value_path)
    assert draft.declared_settings["min_value"]["default"] == 12
    assert load_yaml(draft.to_yaml())["steps"][0]["with"]["min_value"] == 12


def test_invalid_default_and_incompatible_binding_leave_definition_unchanged() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("with.steps.0.with.min_value")
    before = draft.to_yaml()
    with pytest.raises(InvalidError):
        draft.update_declared_setting("min_value", {"default": "bad"})
    assert draft.to_yaml() == before
    with pytest.raises(EditingError, match=re.escape("Setting min_value can't fill Name.")):
        draft.bind("with.steps.0.name", "min_value")
    assert draft.to_yaml() == before


def test_structured_editing_accepts_bindings_and_ordinary_conditions_and_feedback() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("with.steps.0.with.min_value")
    draft.steps_draft.update("with.steps.0", {"if": "{{ answer > 0 }}", "feedback": "{{ answer }}"})
    assert draft.uneditable_reasons == []


@pytest.mark.parametrize(
    ("steps_yaml", "message"),
    [
        ("'{{ settings.checks }}'", "A list of steps is written as a template."),
        (
            "[{uses: gradeflow/full-marks@1, for_each: '{{ settings.items }}'}]",
            "A step repeats for each item of a setting.",
        ),
        (
            "[{uses: gradeflow/number-range@1, with: {min_value: '{{ settings.a + 1 }}'}}]",
            "From is written as a template.",
        ),
        (
            "[{uses: gradeflow/accepted-answers@1, with: {answers: ['{{ settings.a }}']}}]",
            "Accepted answers is written as a template.",
        ),
        (
            "[{uses: gradeflow/key-point@1, with: {point: a, colour: red}}]",
            "Key Point has no setting colour.",
        ),
    ],
)
def test_structured_editing_rejects_authored_structures_without_rewriting(
    steps_yaml: str, message: str
) -> None:
    draft = MethodDraft.from_yaml(
        f"label: Test\nsummary: Test.\nsteps: {steps_yaml}\n",
        NUMBER_QUESTIONS,
        name="test",
        version=1,
    )
    before = draft.to_yaml()
    assert message in draft.uneditable_reasons
    assert draft.to_yaml() == before


def test_default_changes_affect_marking_until_a_test_value_is_explicitly_overridden() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("with.steps.0.with.min_value")

    def twelve_is_correct() -> bool:
        [row] = mark_rows(
            draft.finalized().ref,
            questions=draft.questions,
            question="Q1",
            settings=draft.test_values,
            rows=[ExampleRow(answers={"Q1": "12"})],
            methods=Methods([*BUILTINS, draft.finalized()]),
        )
        return row.results["Q1"].correct

    assert twelve_is_correct()
    draft.steps_draft.update("with.steps.0", {"min_value": 15})
    assert not twelve_is_correct()
    draft.test_values = {"min_value": 5}
    assert twelve_is_correct()
    draft.steps_draft.update("with.steps.0", {"min_value": 18})
    assert twelve_is_correct()
    draft.test_values = {}
    assert not twelve_is_correct()


def test_setting_choices_apply_to_defaults_edited_from_either_location() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("with.marks_by")
    draft.update_declared_setting(
        "marks_by", {"default": "all_correct", "choices": ["all_correct", "any_correct"]}
    )
    with pytest.raises(InvalidError, match="choices"):
        draft.update_declared_setting(
            "marks_by", {"default": "proportional", "choices": ["all_correct", "any_correct"]}
        )
    draft.steps_draft.update("", {"marks_by": "proportional"})
    assert draft.declared_settings["marks_by"]["default"] == "proportional"
    with pytest.raises(EditingError):
        draft.finalized()
    draft.update_declared_setting(
        "marks_by", {"default": "proportional", "choices": ["proportional"]}
    )
    assert draft.steps_draft.values("")["marks_by"] == "proportional"


def test_an_invalid_field_value_cant_become_a_settings_default() -> None:
    draft = number_range_draft()
    draft.steps_draft.update("with.steps.0", {"min_value": "abc"})
    before = draft.to_yaml()
    with pytest.raises(EditingError, match="number"):
        draft.bind_new_setting("with.steps.0.with.min_value")
    assert draft.to_yaml() == before


def test_a_draft_describes_and_validates_bound_defaults_without_materializing_them() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("with.steps.0.with.min_value")
    assert draft.steps_draft.problems == []
    assert "10" in draft.steps_draft.describe("with.steps.0")
    draft.update_declared_setting("min_value", {"default": 11})
    assert "11" in draft.steps_draft.describe("with.steps.0")
    assert load_yaml(draft.to_yaml())["steps"][0]["with"]["min_value"] == "{{ settings.min_value }}"


def test_setting_default_schema_remains_typed_when_an_unrelated_literal_is_invalid() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("with.steps.0.with.min_value")
    draft.bind_new_setting("with.marks_by")
    draft.update_declared_setting(
        "marks_by", {"default": "all_correct", "choices": ["all_correct", "any_correct"]}
    )
    before = draft.declared_setting_schema("min_value")["properties"]["default"]
    draft.steps_draft.update("with.steps.0", {"max_value": "bad"})
    with pytest.raises(EditingError, match="number"):
        _ = draft.settings_schema
    assert draft.declared_setting_schema("min_value")["properties"]["default"] == before
    marks_by_properties = draft.declared_setting_schema("marks_by")["properties"]
    assert marks_by_properties["default"]["enum"] == marks_by_properties["choices"]["items"]["enum"]
    assert marks_by_properties["default"]["enum"] == [
        "all_correct",
        "any_correct",
        "none_correct",
        "total",
        "proportional",
    ]
    with pytest.raises(InvalidError, match="choices"):
        draft.update_declared_setting(
            "marks_by", {"default": "proportional", "choices": ["all_correct", "any_correct"]}
        )
    draft.update_declared_setting("min_value", {"title": "Minimum"})
    assert "default" not in draft.declared_setting_schema("min_value")["properties"]["default"]


def test_bound_parameters_can_be_incomplete_until_the_new_parameter_is_filled() -> None:
    questions = QuestionSet.load({"Q1": {"type": "number"}})
    draft = MethodDraft.new(questions, name="code", version=1, label="Code")
    draft.steps_draft.add("with.steps.0", "gradeflow/custom-code@1")
    draft.steps_draft.update(
        "with.steps.0", {"parameters": {"limit": {"type": "integer", "value": 5}}}
    )
    draft.bind_new_setting("with.steps.0.with.parameters")
    draft.steps_draft.update(
        "with.steps.0",
        {"parameters": {"limit": {"type": "integer", "value": 5}, "new": {"type": "integer"}}},
    )
    draft = MethodDraft.from_yaml(draft.to_yaml(), questions, name="code", version=1)
    assert draft.uneditable_reasons == []
    assert draft.declared_settings["parameters"]["default"]["new"] == {"type": "integer"}
    assert draft.steps_draft.problems == []
    assert draft.problems
    with pytest.raises(EditingError, match="Required"):
        draft.finalized()
    draft.steps_draft.update(
        "with.steps.0",
        {
            "parameters": {
                "limit": {"type": "integer", "value": 5},
                "new": {"type": "integer", "value": 7},
            }
        },
    )
    reopened = MethodDraft.from_yaml(draft.to_yaml(), questions, name="code", version=1)
    assert reopened.steps_draft.values("with.steps.0")["parameters"]["new"]["value"] == 7
    assert reopened.declared_settings["parameters"]["default"]["new"]["value"] == 7
    assert reopened.steps_draft.problems == []
    assert reopened.problems == []
    assert reopened.finalized().ref == "code@1"


def test_a_settings_examples_are_typed_like_the_field_it_fills() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("with.steps.0.with.min_value")
    properties = draft.declared_setting_schema("min_value")["properties"]
    assert properties["examples"]["items"] == {"type": "number"}
    assert "choices" not in properties
    with pytest.raises(InvalidError) as exc_info:
        draft.update_declared_setting("min_value", {"examples": [1, "ten"]})
    assert exc_info.value.problems == [Problem("Must be a number.", field=("examples", 1))]
    draft.update_declared_setting("min_value", {"examples": [10, 2.5]})
    assert draft.declared_settings["min_value"]["examples"] == [10, 2.5]


def test_removing_a_setting_a_check_reads_is_a_problem_at_once() -> None:
    draft = number_range_draft()
    draft.steps_draft.update("with.steps.0", {"feedback": "{{ settings.note }}"})
    draft.update_declared_setting("note", {"title": "Note", "default": "hi"})
    assert draft.steps_draft.problems == []
    draft.remove_declared_setting("note")
    assert [problem.message for problem in draft.steps_draft.problems] == ["settings has no note."]
