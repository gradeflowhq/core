"""Declared setting defaults are authored once, independently of test values."""

import re

import pytest

from gradeflow_core import BUILTINS, InvalidError, Methods, Problem, QuestionSet
from gradeflow_core.editing import EditingError, ExampleRow, MethodDraft, load_yaml, mark_rows

from .helpers import NUMBER_QUESTIONS, number_range_draft


def test_binding_a_new_setting_inherits_current_value_without_creating_a_test_value() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    assert draft.declared_settings["min_value"]["default"] == 10
    assert draft.test_values == {}
    draft.update_declared_setting("min_value", {"title": "Minimum", "default": 12})
    assert draft.values("/with/steps/0")["min_value"] == 12
    draft.update("/with/steps/0", {"min_value": 14})
    assert draft.declared_settings["min_value"]["default"] == 14
    assert load_yaml(draft.to_yaml())["steps"][0]["with"]["min_value"] == "{{ settings.min_value }}"


def test_binding_a_field_of_a_step_written_without_settings_gives_it_them() -> None:
    data = {"label": "In range", "summary": "", "steps": [{"uses": "gradeflow/number-range@1"}]}
    draft = MethodDraft(data, NUMBER_QUESTIONS, name="range", version=1)
    draft.bind_new_setting("/with/steps/0/with/min_value")
    assert load_yaml(draft.to_yaml())["steps"][0]["with"] == {
        "min_value": "{{ settings.min_value }}"
    }


def test_a_condition_may_read_the_methods_settings() -> None:
    draft = number_range_draft()
    assert "if" not in draft.schema("/with/steps/0")["properties"]
    draft.bind_new_setting("/with/steps/0/with/min_value")
    assert "if" in draft.schema("/with/steps/0")["properties"]
    draft.update("/with/steps/0", {"if": "{{ settings.min_value > 0 }}"})
    assert draft.step_problems == []
    assert draft.problems == []


def test_test_values_never_replace_authored_defaults_or_fixed_values() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.test_values = {"min_value": 99}
    assert draft.values("/with/steps/0")["min_value"] == 10
    draft.update("/with/steps/0", {"min_value": 12})
    assert draft.test_values == {"min_value": 99}
    draft.unbind("/with/steps/0/with/min_value")
    assert draft.values("/with/steps/0")["min_value"] == 12
    assert draft.test_values == {}


def test_a_setting_with_no_default_is_checked_with_its_test_value() -> None:
    questions = QuestionSet.load({"Q1": {"type": "text"}})
    draft = MethodDraft.new(questions, name="answers", version=1, label="Answers")
    draft.add("/with/steps/0", "gradeflow/accepted-answers@1")
    draft.bind_new_setting("/with/steps/0/with/answers")
    assert draft.step_problems == []
    assert [str(problem) for problem in draft.problems] == [
        "Give “Accepted answers” a test value or a default."
    ]
    draft.add("/with/steps/1", "gradeflow/equivalent-expression@1")
    draft.update("/with/steps/1", {"answers": ["x2"]})
    expected = [Problem("Unknown name “x2”.", "/with/steps/1", "/with/answers/0")]
    assert draft.step_problems == expected, "while the first step waits for its setting"
    draft.test_values = {"answers": ["sun"]}
    assert draft.step_problems == expected
    assert draft.problems == expected
    assert draft.values("/with/steps/0")["answers"] is None


@pytest.mark.parametrize("value", [None, "", [], {}])
def test_empty_values_create_required_settings(value: object) -> None:
    draft = number_range_draft()
    marking = draft.data
    marking["with"]["steps"][0]["with"]["min_value"] = value
    draft.replace(marking)
    draft.bind_new_setting("/with/steps/0/with/min_value")
    assert "default" not in draft.declared_settings["min_value"]
    assert draft.values("/with/steps/0")["min_value"] is None


def test_zero_and_false_are_defaults_and_clearing_a_default_is_required() -> None:
    draft = number_range_draft()
    draft.update("/with/steps/0", {"min_value": 0})
    draft.bind_new_setting("/with/steps/0/with/min_value")
    assert draft.declared_settings["min_value"]["default"] == 0
    draft.update("/with/steps/0", {"min_value": None})
    assert "default" not in draft.declared_settings["min_value"]
    text_draft = MethodDraft.new(
        QuestionSet.load({"Q1": {"type": "text"}}), name="text", version=1, label="Text"
    )
    text_draft.add("/with/steps/0", "gradeflow/key-points@1")
    text_draft.bind_new_setting("/with/steps/0/with/whole_words")
    assert text_draft.declared_settings["whole_words"]["default"] is False


def test_binding_existing_setting_preserves_default_and_updates_every_bound_field() -> None:
    draft = number_range_draft()
    min_value_path = "/with/steps/0/with/min_value"
    max_value_path = "/with/steps/0/with/max_value"
    draft.bind_new_setting(min_value_path)
    draft.bind_new_setting(max_value_path)
    assert "min_value" in draft.compatible_settings(max_value_path)
    draft.bind(max_value_path, "min_value")
    assert set(draft.declared_settings) == {"min_value"}
    assert draft.values("/with/steps/0")["max_value"] == 10
    # An update with both fields; the unchanged sibling must not undo the edit.
    draft.update("/with/steps/0", {"min_value": 12, "max_value": 10})
    assert draft.values("/with/steps/0")["min_value"] == 12
    assert draft.values("/with/steps/0")["max_value"] == 12
    draft.unbind(min_value_path)
    assert draft.declared_settings["min_value"]["default"] == 12
    assert load_yaml(draft.to_yaml())["steps"][0]["with"]["min_value"] == 12


def test_invalid_default_and_incompatible_binding_leave_definition_unchanged() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    before = draft.to_yaml()
    with pytest.raises(InvalidError):
        draft.update_declared_setting("min_value", {"default": "bad"})
    assert draft.to_yaml() == before
    with pytest.raises(EditingError, match=re.escape("Setting min_value can't fill Name.")):
        draft.bind("/with/steps/0/name", "min_value")
    assert draft.to_yaml() == before


def test_structured_editing_accepts_bindings_and_ordinary_conditions_and_feedback() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.update("/with/steps/0", {"if": "{{ answer > 0 }}", "feedback": "{{ answer }}"})
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
    draft.bind_new_setting("/with/steps/0/with/min_value")

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
    draft.update("/with/steps/0", {"min_value": 15})
    assert not twelve_is_correct()
    draft.test_values = {"min_value": 5}
    assert twelve_is_correct()
    draft.update("/with/steps/0", {"min_value": 18})
    assert twelve_is_correct()
    draft.test_values = {}
    assert not twelve_is_correct()


def test_setting_choices_apply_to_defaults_edited_from_either_location() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/marks_by")
    draft.update_declared_setting(
        "marks_by", {"default": "all_correct", "choices": ["all_correct", "any_correct"]}
    )
    with pytest.raises(InvalidError, match="choices"):
        draft.update_declared_setting(
            "marks_by", {"default": "proportional", "choices": ["all_correct", "any_correct"]}
        )
    draft.update("", {"marks_by": "proportional"})
    assert draft.declared_settings["marks_by"]["default"] == "proportional"
    with pytest.raises(EditingError):
        draft.finalized()
    draft.update_declared_setting(
        "marks_by", {"default": "proportional", "choices": ["proportional"]}
    )
    assert draft.values("")["marks_by"] == "proportional"


def test_an_invalid_field_value_cant_become_a_settings_default() -> None:
    draft = number_range_draft()
    draft.update("/with/steps/0", {"min_value": "abc"})
    before = draft.to_yaml()
    with pytest.raises(EditingError, match="number"):
        draft.bind_new_setting("/with/steps/0/with/min_value")
    assert draft.to_yaml() == before


def test_a_draft_describes_and_validates_bound_defaults_without_materializing_them() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    assert draft.step_problems == []
    assert "10" in draft.describe("/with/steps/0")
    draft.update_declared_setting("min_value", {"default": 11})
    assert "11" in draft.describe("/with/steps/0")
    assert load_yaml(draft.to_yaml())["steps"][0]["with"]["min_value"] == "{{ settings.min_value }}"


def test_setting_default_schema_remains_typed_when_an_unrelated_literal_is_invalid() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.bind_new_setting("/with/marks_by")
    draft.update_declared_setting(
        "marks_by", {"default": "all_correct", "choices": ["all_correct", "any_correct"]}
    )
    before = draft.declared_setting_schema("min_value")["properties"]["default"]
    draft.update("/with/steps/0", {"max_value": "bad"})
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


def test_removing_a_setting_a_check_reads_is_a_problem_at_once() -> None:
    draft = number_range_draft()
    draft.update("/with/steps/0", {"feedback": "{{ settings.note }}"})
    draft.update_declared_setting("note", {"title": "Note", "default": "hi"})
    assert draft.step_problems == []
    draft.remove_declared_setting("note")
    assert [problem.message for problem in draft.step_problems] == ["settings has no note."]


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("weight", "Steps already have a field called weight. Choose another name."),
        ("expansion", "A setting can't be called expansion. Choose another name."),
        ("_hidden", "A setting can't be called _hidden. Choose another name."),
    ],
)
def test_a_setting_cant_take_a_name_steps_or_settings_have(name: str, message: str) -> None:
    draft = number_range_draft()
    with pytest.raises(EditingError, match=re.escape(message)):
        draft.update_declared_setting(name, {"title": "Taken"})
    assert name not in draft.declared_settings


def code_draft() -> MethodDraft:
    """A method whose code reads its setting ``minimum``."""
    draft = MethodDraft.new(NUMBER_QUESTIONS, name="code", version=1, label="Code")
    draft.add("/with/steps/0", "gradeflow/custom-code@1")
    draft.update("/with/steps/0", {"code": 'correct = answer >= settings["minimum"]'})
    draft.update_declared_setting("minimum", {"title": "Minimum", "type": "number", "default": 3})
    return draft


def test_code_reads_a_typed_setting_when_the_method_is_tried() -> None:
    draft = code_draft()
    assert draft.problems == []
    assert draft.declared_setting_schema("minimum")["required"] == ["type"]
    method = draft.finalized()
    rows = mark_rows(
        method.ref,
        questions=draft.questions,
        question="Q1",
        settings={},
        rows=[ExampleRow(answers={"Q1": "2"}), ExampleRow(answers={"Q1": "5"})],
        methods=Methods([*BUILTINS, method]),
    )
    assert [row.results["Q1"].correct for row in rows] == [False, True]


def test_a_setting_without_a_type_or_use_is_a_problem() -> None:
    draft = code_draft()
    draft.update_declared_setting("minimum", {"title": "Minimum", "default": 3})
    assert [problem.message for problem in draft.problems] == ["Minimum isn't used."]


def test_a_setting_bound_to_a_field_has_no_type_to_choose() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    assert "type" not in draft.declared_setting_schema("min_value")["properties"]


def test_an_added_setting_takes_its_default_as_its_type_says() -> None:
    draft = number_range_draft()
    draft.add_declared_setting("minimum")
    assert "default" not in draft.declared_setting_schema("minimum")["properties"]
    draft.update_declared_setting("minimum", {"title": "Minimum", "type": "integer"})
    assert draft.declared_setting_schema("minimum")["properties"]["default"]["type"] == "integer"
    assert draft.problems == [Problem("Give “Minimum” a test value or a default.")]
    with pytest.raises(EditingError, match="already a setting called minimum"):
        draft.add_declared_setting("minimum")


def test_code_in_a_method_is_told_its_settings_and_their_types() -> None:
    draft = code_draft()
    description = draft.schema("/with/steps/0")["properties"]["code"]["description"]
    assert '`settings["minimum"]` (float)' in description


@pytest.mark.parametrize(
    ("values", "problem"),
    [
        ({"type": "number_list", "default": 3}, Problem("Must be a list.", field="/default")),
        (
            {"type": "number_list", "default": [1, "a"]},
            Problem("Must be a number.", field="/default"),
        ),
        ({"type": "integer", "default": 2.5}, Problem("Must be a whole number.", field="/default")),
        ({"type": "number", "default": "inf"}, Problem("Must be a number.", field="/default")),
    ],
)
def test_a_setting_takes_only_values_of_its_type(
    values: dict[str, object], problem: Problem
) -> None:
    draft = code_draft()
    with pytest.raises(InvalidError) as error:
        draft.update_declared_setting("minimum", values)
    assert error.value.problems == [problem]
    assert draft.declared_settings["minimum"]["type"] == "number"
