"""Input defaults are authored once, independently of representative test overrides."""

import re

import pytest

from gradeflow_core import BUILTINS, InvalidError, Methods, Problem, QuestionSet
from gradeflow_core.editing import EditingError, ExampleRow, MethodDraft, evaluate_rows, load_yaml

NUMBER = QuestionSet.model_validate({"Q1": {"type": "number"}})


def ranged() -> MethodDraft:
    draft = MethodDraft.new(NUMBER, name="range", version=1, label="Range")
    draft.checks.add("with.checks.0", "gradeflow/number-range@1")
    draft.checks.update("with.checks.0", {"min_value": 10, "max_value": 20})
    return draft


def test_exposing_inherits_current_value_without_creating_test_override() -> None:
    draft = ranged()
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    assert draft.inputs["min_value"]["default"] == 10
    assert draft.test_values == {}
    draft.update_input("min_value", {"title": "Minimum", "default": 12})
    assert draft.checks.values("with.checks.0")["min_value"] == 12
    draft.checks.update("with.checks.0", {"min_value": 14})
    assert draft.inputs["min_value"]["default"] == 14
    assert load_yaml(draft.to_yaml())["checks"][0]["with"]["min_value"] == "{{ inputs.min_value }}"


def test_a_condition_may_read_the_methods_inputs() -> None:
    draft = ranged()
    assert "if" not in draft.checks.schema("with.checks.0")["properties"]
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    assert "if" in draft.checks.schema("with.checks.0")["properties"]
    draft.checks.update("with.checks.0", {"if": "{{ inputs.min_value > 0 }}"})
    assert draft.checks.problems == []
    assert draft.problems == []


def test_test_overrides_never_replace_authored_defaults_or_fixed_values() -> None:
    draft = ranged()
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    draft.test_values = {"min_value": 99}
    assert draft.checks.values("with.checks.0")["min_value"] == 10
    draft.checks.update("with.checks.0", {"min_value": 12})
    assert draft.test_values == {"min_value": 99}
    draft.fix("with.checks.0.with.min_value")
    assert draft.checks.values("with.checks.0")["min_value"] == 12
    assert draft.test_values == {}


@pytest.mark.parametrize("value", [None, "", [], {}])
def test_empty_values_create_required_inputs(value: object) -> None:
    draft = ranged()
    data = draft.checks.data
    data["with"]["checks"][0]["with"]["min_value"] = value
    draft.checks.replace(data)
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    assert "default" not in draft.inputs["min_value"]
    assert draft.checks.values("with.checks.0")["min_value"] is None


def test_zero_and_false_are_defaults_and_clearing_a_default_is_required() -> None:
    draft = ranged()
    draft.checks.update("with.checks.0", {"min_value": 0})
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    assert draft.inputs["min_value"]["default"] == 0
    draft.checks.update("with.checks.0", {"min_value": None})
    assert "default" not in draft.inputs["min_value"]
    text = MethodDraft.new(
        QuestionSet.model_validate({"Q1": {"type": "text"}}), name="text", version=1, label="Text"
    )
    text.checks.add("with.checks.0", "gradeflow/key-points@1")
    text.expose("with.checks.0.with.whole_words", label="Whole words")
    assert text.inputs["whole_words"]["default"] is False


def test_binding_existing_input_preserves_default_and_updates_every_bound_field() -> None:
    draft = ranged()
    first = "with.checks.0.with.min_value"
    second = "with.checks.0.with.max_value"
    draft.expose(first, label="Limit")
    draft.expose(second, label="Maximum")
    assert "min_value" in draft.compatible_inputs(second)
    draft.bind(second, "min_value")
    assert set(draft.inputs) == {"min_value"}
    assert draft.checks.values("with.checks.0")["max_value"] == 10
    # An update with both fields; the unchanged sibling must not undo the edit.
    draft.checks.update("with.checks.0", {"min_value": 12, "max_value": 10})
    assert draft.checks.values("with.checks.0")["min_value"] == 12
    assert draft.checks.values("with.checks.0")["max_value"] == 12
    draft.fix(first)
    assert draft.inputs["min_value"]["default"] == 12
    assert load_yaml(draft.to_yaml())["checks"][0]["with"]["min_value"] == 12


def test_invalid_default_and_incompatible_binding_leave_definition_unchanged() -> None:
    draft = ranged()
    draft.expose("with.checks.0.with.min_value", label="Limit")
    before = draft.to_yaml()
    with pytest.raises(InvalidError):
        draft.update_input("min_value", {"default": "bad"})
    assert draft.to_yaml() == before
    with pytest.raises(EditingError, match=re.escape("Input min_value can't fill Name.")):
        draft.bind("with.checks.0.name", "min_value")
    assert draft.to_yaml() == before


def test_structured_editing_accepts_bindings_and_ordinary_conditions_and_feedback() -> None:
    draft = ranged()
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    draft.checks.update("with.checks.0", {"if": "{{ answer > 0 }}", "feedback": "{{ answer }}"})
    assert draft.uneditable == []


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        ("'{{ inputs.checks }}'", "A check list is an expression."),
        (
            "[{uses: gradeflow/full-marks@1, for_each: '{{ inputs.items }}'}]",
            "A check repeats for each item of an input.",
        ),
        (
            "[{uses: gradeflow/number-range@1, with: {min_value: '{{ inputs.a + 1 }}'}}]",
            "From is an expression.",
        ),
        (
            "[{uses: gradeflow/accepted-answers@1, with: {answers: ['{{ inputs.a }}']}}]",
            "Accepted answers is an expression.",
        ),
        (
            "[{uses: gradeflow/key-point@1, with: {point: a, colour: red}}]",
            "Key Point has no setting colour.",
        ),
    ],
)
def test_structured_editing_rejects_authored_structures_without_rewriting(
    checks: str, message: str
) -> None:
    draft = MethodDraft.from_yaml(
        f"label: Test\nsummary: Test.\nchecks: {checks}\n", NUMBER, name="test", version=1
    )
    before = draft.to_yaml()
    assert message in draft.uneditable
    assert draft.to_yaml() == before


def test_default_changes_affect_grading_until_a_test_value_is_explicitly_overridden() -> None:
    draft = ranged()
    draft.expose("with.checks.0.with.min_value", label="Minimum")

    def correct() -> bool:
        [row] = evaluate_rows(
            draft.ref,
            questions=draft.questions,
            question="Q1",
            inputs=draft.test_values,
            rows=[ExampleRow(answers={"Q1": "12"})],
            methods=Methods([*BUILTINS, draft.finalized()]),
        )
        return row.results["Q1"].correct

    assert correct()
    draft.checks.update("with.checks.0", {"min_value": 15})
    assert not correct()
    draft.test_values = {"min_value": 5}
    assert correct()
    draft.checks.update("with.checks.0", {"min_value": 18})
    assert correct()
    draft.test_values = {}
    assert not correct()


def test_input_choices_apply_to_defaults_edited_from_either_location() -> None:
    draft = ranged()
    draft.expose("with.combine", label="Combine")
    draft.update_input("combine", {"default": "all", "choices": ["all", "any"]})
    with pytest.raises(InvalidError, match="choices"):
        draft.update_input("combine", {"default": "proportional", "choices": ["all", "any"]})
    draft.checks.update("", {"combine": "proportional"})
    assert draft.inputs["combine"]["default"] == "proportional"
    with pytest.raises(EditingError):
        draft.finalized()
    draft.update_input("combine", {"default": "proportional", "choices": ["proportional"]})
    assert draft.checks.values("")["combine"] == "proportional"


def test_a_draft_describes_and_validates_bound_defaults_without_materializing_them() -> None:
    draft = ranged()
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    assert draft.checks.problems == []
    assert "10" in draft.checks.describe("with.checks.0")
    draft.update_input("min_value", {"default": 11})
    assert "11" in draft.checks.describe("with.checks.0")
    assert load_yaml(draft.to_yaml())["checks"][0]["with"]["min_value"] == "{{ inputs.min_value }}"


def test_input_default_schema_remains_typed_when_an_unrelated_literal_is_invalid() -> None:
    draft = ranged()
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    draft.expose("with.combine", label="Combine")
    draft.update_input("combine", {"default": "all", "choices": ["all", "any"]})
    before = draft.input_schema("min_value")["properties"]["default"]
    draft.checks.update("with.checks.0", {"max_value": "bad"})
    with pytest.raises(EditingError, match="number"):
        _ = draft.settings_schema
    assert draft.input_schema("min_value")["properties"]["default"] == before
    combine = draft.input_schema("combine")["properties"]
    assert combine["default"]["enum"] == combine["choices"]["items"]["enum"]
    assert combine["default"]["enum"] == ["all", "any", "none", "total", "proportional"]
    with pytest.raises(InvalidError, match="choices"):
        draft.update_input("combine", {"default": "proportional", "choices": ["all", "any"]})
    draft.update_input("min_value", {"title": "Minimum"})
    assert "default" not in draft.input_schema("min_value")["properties"]["default"]


def test_bound_collection_can_be_incomplete_until_its_new_entry_is_filled() -> None:
    questions = QuestionSet.model_validate({"Q1": {"type": "number"}})
    draft = MethodDraft.new(questions, name="code", version=1, label="Code")
    draft.checks.add("with.checks.0", "gradeflow/custom-code@1")
    draft.checks.update("with.checks.0", {"parameters": {"limit": {"type": "integer", "value": 5}}})
    draft.expose("with.checks.0.with.parameters", label="Parameters")
    draft.checks.update(
        "with.checks.0",
        {"parameters": {"limit": {"type": "integer", "value": 5}, "new": {"type": "integer"}}},
    )
    draft = MethodDraft.from_yaml(draft.to_yaml(), questions, name="code", version=1)
    assert draft.uneditable == []
    assert draft.inputs["parameters"]["default"]["new"] == {"type": "integer"}
    assert draft.checks.problems == []
    assert draft.problems
    with pytest.raises(EditingError, match="Required"):
        draft.finalized()
    draft.checks.update(
        "with.checks.0",
        {
            "parameters": {
                "limit": {"type": "integer", "value": 5},
                "new": {"type": "integer", "value": 7},
            }
        },
    )
    reopened = MethodDraft.from_yaml(draft.to_yaml(), questions, name="code", version=1)
    assert reopened.checks.values("with.checks.0")["parameters"]["new"]["value"] == 7
    assert reopened.inputs["parameters"]["default"]["new"]["value"] == 7
    assert reopened.checks.problems == []
    assert reopened.problems == []
    assert reopened.finalized().ref == "code@1"


def test_an_inputs_examples_are_typed_like_the_setting_it_fills() -> None:
    draft = ranged()
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    schema = draft.input_schema("min_value")["properties"]
    assert schema["examples"]["items"] == {"type": "number"}
    assert "choices" not in schema
    with pytest.raises(InvalidError) as caught:
        draft.update_input("min_value", {"examples": [1, "ten"]})
    assert caught.value.problems == [Problem("Must be a number.", field=("examples", 1))]
    draft.update_input("min_value", {"examples": [10, 2.5]})
    assert draft.inputs["min_value"]["examples"] == [10, 2.5]


def test_removing_an_input_a_check_reads_is_a_problem_at_once() -> None:
    draft = ranged()
    draft.checks.update("with.checks.0", {"feedback": "{{ inputs.note }}"})
    draft.update_input("note", {"title": "Note", "default": "hi"})
    assert draft.checks.problems == []
    draft.remove_input("note")
    assert [problem.message for problem in draft.checks.problems] == ["inputs has no note."]
