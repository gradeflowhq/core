import re
from pathlib import Path

import pytest

from gradeflow_core import (
    BUILTINS,
    Check,
    GradeFlowError,
    InvalidError,
    MethodFileError,
    Methods,
    Problem,
    QuestionSet,
)
from gradeflow_core.editing import (
    HINT_KEY,
    Condition,
    ConditionRow,
    EditingError,
    ExampleRow,
    ExpectedResult,
    MethodDraft,
    RowResult,
    evaluate_rows,
    load_yaml,
)
from gradeflow_core.methods import catalogue

NUMBER = QuestionSet.model_validate({"Q1": {"type": "number", "max_marks": 2}})
PAIR = QuestionSet.model_validate({"Q1": {"type": "number"}, "Q2": {"type": "number"}})
TEXT = QuestionSet.model_validate({"Q1": {"type": "text"}})


def numeric() -> MethodDraft:
    draft = MethodDraft.new(
        NUMBER, name="range", version=1, label="Within a range", summary="In range."
    )
    draft.checks.add("with.checks.0", "gradeflow/number-range@1")
    draft.checks.update("with.checks.0", {"min_value": 10, "max_value": 20})
    return draft


def reopen(draft: MethodDraft, *, version: int = 1) -> MethodDraft:
    return MethodDraft.from_yaml(draft.to_yaml(), draft.questions, name=draft.name, version=version)


def said(draft: MethodDraft) -> list[str]:
    return [str(problem) for problem in draft.problems]


def rows(*answers: str) -> list[ExampleRow]:
    return [ExampleRow(answers={"Q1": answer}) for answer in answers]


def tried(
    draft: MethodDraft, examples: list[ExampleRow], methods: Methods = BUILTINS
) -> list[RowResult]:
    return evaluate_rows(
        draft.ref,
        questions=draft.questions,
        question=draft.question,
        inputs=draft.test_values,
        rows=examples,
        methods=Methods([*methods, draft.finalized()]),
    )


def test_a_new_draft_is_an_empty_combination_written_with_its_version_first() -> None:
    draft = MethodDraft.new(NUMBER, name="empty", version=3, label="Empty")
    assert draft.ref == "empty@3"
    assert draft.checks.data == {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {"checks": []},
    }
    assert list(load_yaml(draft.to_yaml())) == [
        "version",
        "label",
        "summary",
        "question_types",
        "checks",
    ]
    assert load_yaml(draft.to_yaml())["version"] == 3


def test_a_files_checks_are_edited_as_their_combination_which_keeps_its_method() -> None:
    one = reopen(numeric())
    assert one.checks.data == numeric().checks.data
    assert one.to_yaml() == numeric().to_yaml()
    assert [method.ref for method in one.checks.methods("")] == ["gradeflow/combination@1"]
    assert not {"name", "id", "if", "feedback"} & one.checks.schema("")["properties"].keys()

    one.checks.add("with.checks.1", "gradeflow/full-marks@1")
    assert reopen(one).checks.is_check("with.checks.1")

    summed = numeric()
    summed.checks.update("", {"combine": "proportional"})
    assert reopen(summed).checks.data["with"]["combine"] == "proportional"


def test_bound_settings_follow_moved_checks_and_examples_never_change_defaults() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Lower limit", default=10)
    draft.expose("with.checks.0.with.max_value", label="Upper limit", description="Inclusive.")
    draft.checks.add("with.checks.1", "gradeflow/full-marks@1")
    draft.checks.move("with.checks.0", 1)
    text = draft.to_yaml()
    document = load_yaml(text)
    assert document["version"] == 1
    assert document["checks"][1]["with"]["min_value"] == "{{ inputs.min_value }}"
    assert document["inputs"]["min_value"]["default"] == 10
    assert document["inputs"]["max_value"]["default"] == 20
    reopened = MethodDraft.from_yaml(
        text, draft.questions, name="range", version=1, test_values={"max_value": 25}
    )
    assert reopened.checks.values("with.checks.1")["max_value"] == 20
    assert load_yaml(reopened.to_yaml())["inputs"]["max_value"]["default"] == 20
    assert reopened.settings_schema["properties"]["min_value"]["title"] == "Lower limit"
    later = reopen(reopened, version=4)
    assert BUILTINS.with_files([("sample/range", later.to_yaml())])[-1].ref == "sample/range@4"


def test_deletion_and_retyping_report_affected_inputs_and_allow_explicit_removal() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Lower limit", default=10)
    draft.checks.remove("with.checks.0")
    assert "Lower limit isn't used." in said(draft)
    with pytest.raises(EditingError, match="Lower limit"):
        draft.finalized()
    draft.remove_input("min_value")
    assert "inputs" not in load_yaml(draft.to_yaml())


def test_repeated_yaml_stays_executable_and_requires_whole_document_editing() -> None:
    text = """label: Repeated
summary: Repeated checks.
question_types: [text]
inputs:
  terms: {default: [a, b]}
checks:
  - for_each: '{{ inputs.terms }}'
    uses: gradeflow/key-point@1
    with: {point: '{{ each }}'}
"""
    draft = MethodDraft.from_yaml(text, TEXT, name="repeated", version=1)
    assert not [problem for problem in said(draft) if "isn't used" in problem]
    assert load_yaml(draft.to_yaml()) == {"version": 1, **load_yaml(text)}
    assert draft.uneditable == ["A check repeats for each item of an input."]
    assert [row.results["Q1"].correct for row in tried(draft, rows("a b", "c"))] == [True, False]


def test_compatibility_refuses_a_number_check_declared_for_text() -> None:
    draft = numeric()
    draft.update_metadata(question_types=["number", "text"])
    assert any(problem.startswith("It can't mark text questions: ") for problem in said(draft))
    with pytest.raises(EditingError, match="It can't mark text questions"):
        draft.finalized()


def test_template_only_inputs_are_used_even_without_visual_setting_fields() -> None:
    text = """label: Conditional
summary: Optional check.
question_types: [number]
inputs:
  enabled: {default: true}
  message: {default: Good.}
  unused: {default: 10}
checks:
  - uses: gradeflow/full-marks@1
    if: '{{ inputs.enabled }}'
feedback: '{{ inputs.message }}'
"""
    draft = MethodDraft.from_yaml(text, NUMBER, name="conditional", version=1)
    assert [problem for problem in said(draft) if "isn't used" in problem] == [
        "Input unused isn't used."
    ]


def test_trying_rows_keeps_scratch_and_expectations_separate() -> None:
    draft = numeric()
    expected = {"Q1": ExpectedResult(marks=2, feedback="Correct.")}
    results = tried(
        draft,
        [
            ExampleRow(answers={"Q1": "15"}),
            ExampleRow(answers={"Q1": "8"}, expected=expected),
            ExampleRow(answers={"Q1": "invalid"}),
            ExampleRow(answers={"Q1": ""}),
        ],
    )
    assert (results[0].total_marks, results[0].matches) == (2, None)
    assert (results[1].total_marks, results[1].matches) == (0, False)
    assert not results[2].results["Q1"].marked
    assert results[3].total_marks == 0
    assert draft.examples == ()


def test_versions_are_explicit_positive_and_written_by_the_draft() -> None:
    text = numeric().to_yaml()
    for invalid, message in ((0, "Must be more than 0"), (-1, "Must be more than 0")):
        draft = MethodDraft.from_yaml(text, NUMBER, name="range", version=invalid)
        with pytest.raises(EditingError, match=message):
            draft.finalized()
    with pytest.raises(MethodFileError, match="version"):
        BUILTINS.with_files([("sample/range", text.replace("version: 1\n", ""))])
    fifth = MethodDraft.from_yaml(text, NUMBER, name="range", version=5)
    assert fifth.to_yaml() == text.replace("version: 1", "version: 5")
    assert MethodDraft.from_yaml(fifth.to_yaml(), NUMBER, name="range", version=1).to_yaml() == text
    assert all(method.version >= 1 for method in BUILTINS)


def test_group_question_inputs_are_reusable_and_all_sample_answers_are_required() -> None:
    draft = MethodDraft.new(PAIR, name="pair", version=1, label="Pair", summary="Both.")
    draft.checks.add("with.checks.0", "gradeflow/number-range@1", "Q1")
    draft.checks.add("with.checks.1", "gradeflow/number-range@1", "Q2")
    draft.checks.update("with.checks.0", {"min_value": 10})
    draft.checks.update("with.checks.1", {"max_value": 20})
    draft.expose("with.checks.0.question", label="First question", default="Q1")
    draft.expose("with.checks.1.question", label="Second question", default="Q2")
    [result] = tried(draft, [ExampleRow(answers={"Q1": "15", "Q2": "25"})])
    assert result.total_marks == 1
    assert result.results["Q1"].correct
    assert not result.results["Q2"].correct
    with pytest.raises(GradeFlowError, match=re.escape("Row 1 has no answer to Q2.")):
        tried(draft, rows("15"))


def test_retyping_a_bound_check_refuses_to_rebind_it_to_the_new_method() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=10)
    draft.checks.retype("with.checks.0", "gradeflow/full-marks@1")
    assert said(draft) == ["Minimum isn't used."]
    assert draft.inputs["min_value"]["default"] == 10
    draft.remove_input("min_value")
    assert draft.problems == []


def test_root_retyping_preserves_the_selected_method_and_feedback() -> None:
    draft = numeric()
    draft.checks.retype("", "gradeflow/number-range@1")
    draft.checks.update("", {"min_value": 10, "feedback": "My feedback."})
    [result] = tried(reopen(draft), rows("11"))
    assert (result.total_marks, result.results["Q1"].feedback) == (2, "My feedback.")


def test_from_marking_preserves_templates_written_as_literal_answer_text() -> None:
    check = Check.load(
        {"uses": "gradeflow/key-point@1", "question": "Q1", "with": {"point": "{{ literal }}"}}
    )
    draft = MethodDraft.from_marking(
        check, TEXT, name="braces", version=1, label="Braces", summary="Braces."
    )
    [result] = tried(draft, rows("{{ literal }}"))
    assert result.total_marks == 1


def test_invalid_input_defaults_are_rejected_even_when_test_values_are_valid() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=10)
    document = load_yaml(draft.to_yaml())
    document["inputs"]["min_value"]["default"] = "not a number"
    reopened = MethodDraft(document, NUMBER, name="range", version=1, test_values={"min_value": 10})
    with pytest.raises(EditingError, match="number"):
        reopened.finalized()


def test_fixing_an_exposed_setting_preserves_its_inherited_default() -> None:
    draft = MethodDraft.new(TEXT, name="cat", version=1, label="Cat")
    draft.checks.add("with.checks.0", "gradeflow/key-points@1")
    draft.checks.update("with.checks.0", {"points": ["cat"]})
    draft.expose("with.checks.0.with.whole_words", label="Whole words")
    reopened = reopen(draft)
    assert reopened.checks.values("with.checks.0")["whole_words"] is False
    reopened.fix("with.checks.0.with.whole_words")
    assert reopened.problems == []


def test_opening_a_bound_definition_does_not_change_publishable_content() -> None:
    draft = reopen(numeric(), version=3)
    draft.expose("with.checks.0.with.min_value", label="Lower limit", default=10)
    text = draft.to_yaml()
    assert reopen(draft, version=3).to_yaml() == text


def test_a_new_version_may_use_the_previous_one_but_not_itself() -> None:
    published = numeric().finalized()
    text = numeric().to_yaml()
    edited = MethodDraft.from_yaml(
        text, NUMBER, name="range", version=2, methods=Methods([*BUILTINS, published])
    )
    edited.checks.update("", {"max_value": 25})
    assert edited.finalized().ref == "range@2"
    assert published.version == 1
    edited.checks.retype("", "range@1")
    assert edited.problems == []
    assert [
        row.total_marks for row in tried(edited, rows("15", "25"), Methods([*BUILTINS, published]))
    ] == [2, 0]
    itself = MethodDraft.from_yaml(
        text, NUMBER, name="range", version=1, methods=Methods([*BUILTINS, published])
    )
    itself.checks.retype("", "range@1")
    with pytest.raises(EditingError, match=re.escape("There's no method range@1.")):
        itself.finalized()


def test_exposed_setting_keeps_its_condition_reference_after_reload() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Lower limit", default=10)
    draft.checks.add("with.checks.1", "gradeflow/full-marks@1")
    draft.checks.update(
        "with.checks.1", {"if": Condition((ConditionRow("with.checks.0", "correct"),))}
    )
    reopened = reopen(draft)
    assert reopened.problems == []
    assert isinstance(reopened.checks.values("with.checks.1")["if"], Condition)
    field = next(field for field in reopened.fields() if field.input == "min_value")
    assert (field.check, field.key) == ("with.checks.0", "min_value")


@pytest.mark.parametrize(
    "content",
    [
        "checks: 2",
        "checks: [{uses: gradeflow/full-marks@1, with: bad}]",
    ],
)
def test_incomplete_yaml_remains_a_draft_with_problems(content: str) -> None:
    text = f"label: Incomplete\nsummary: Incomplete.\nquestion_types: [text]\n{content}\n"
    draft = MethodDraft.from_yaml(text, TEXT, name="incomplete", version=1)
    assert draft.to_yaml()
    assert draft.problems


def test_inputs_that_arent_declarations_cant_be_opened() -> None:
    text = "label: Incomplete\nsummary: Incomplete.\ninputs: {a: null}\nchecks: []\n"
    with pytest.raises(
        EditingError, match=re.escape("Inputs must be a mapping of names to their settings.")
    ):
        MethodDraft.from_yaml(text, TEXT, name="incomplete", version=1)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- label: Listed\n", "A method definition must be a mapping."),
        ("label: [\n", "Line 2: expected the node content, but found '<stream end>'."),
    ],
)
def test_text_that_isnt_a_method_file_cant_be_opened(text: str, message: str) -> None:
    with pytest.raises(InvalidError) as caught:
        MethodDraft.from_yaml(text, TEXT, name="incomplete", version=1)
    assert caught.value.problems == [Problem(message)]


def test_question_count_derives_scope_and_preserves_bound_checks() -> None:
    draft = MethodDraft.new(NUMBER, name="range", version=1, label="Range")
    draft.checks.add("with.checks.0", "gradeflow/number-range@1")
    draft.checks.update("with.checks.0", {"min_value": 10})
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=5)
    draft.set_questions(PAIR)
    assert (draft.question, draft.finalized().group) == (None, True)
    assert draft.checks.values("with.checks.0")["question"] == "Q1"
    assert next(field for field in draft.fields() if field.key == "min_value").input == "min_value"
    assert draft.test_values == {}
    draft.set_questions(NUMBER)
    assert draft.question == "Q1"
    assert not draft.finalized().group
    assert "question" not in load_yaml(draft.to_yaml())["checks"][0]
    assert draft.test_values == {}


def test_removed_questions_leave_diagnosable_logic_and_empty_drafts_cannot_publish() -> None:
    draft = MethodDraft.new(PAIR, name="range", version=1, label="Range")
    draft.checks.add("with.checks.0", "gradeflow/number-range@1", "Q2")
    draft.checks.update("with.checks.0", {"min_value": 10})
    draft.set_questions(QuestionSet({"Q1": PAIR["Q1"]}))
    assert draft.checks.values("with.checks.0")["question"] == "Q2"
    assert "Q2" in " ".join(said(draft))
    draft.set_questions(QuestionSet({}))
    assert said(draft) == ["Add at least one question."]
    with pytest.raises(EditingError, match=re.escape("Add at least one question.")):
        draft.finalized()


def test_configurable_values_use_the_current_field_value_as_the_default() -> None:
    draft = MethodDraft.new(TEXT, name="cat", version=1, label="Cat")
    draft.checks.add("with.checks.0", "gradeflow/key-points@1")
    draft.checks.update("with.checks.0", {"points": ["cat"]})
    draft.expose("with.checks.0.with.whole_words", label="Whole words")
    assert draft.test_values == {}
    assert "whole_words" not in draft.settings_schema.get("required", [])
    assert draft.finalized().settings.model_fields["whole_words"].default is False
    draft.expose("with.checks.0.with.whole_words", label="Whole words", default=True)
    assert draft.test_values == {}
    assert draft.finalized().settings.model_fields["whole_words"].default is True
    draft.fix("with.checks.0.with.whole_words")
    assert draft.test_values == {}
    assert not draft.finalized().settings.model_fields


def test_examples_publish_without_execution_or_fabricated_expectations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = numeric()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Publishing examples must not run marking.")

    monkeypatch.setattr("gradeflow_core.marking.examples.mark", forbidden)
    wrong = {"Q1": ExpectedResult(marks=2, feedback="Wrong.")}
    added = draft.set_example(
        [ExampleRow(answers={"Q1": "15"}), ExampleRow(answers={"Q1": "8"}, expected=wrong)],
        at=None,
    )
    assert added == 0
    method = draft.finalized()
    first, second = method.examples[0].rows
    assert (first.answers, first.expected) == ({"Q1": "15"}, {})
    assert second.expected == wrong
    assert method.examples[0].questions == dict(draft.questions)
    assert draft.set_example([], at=0) is None
    assert draft.finalized().examples == ()


def test_changed_test_values_do_not_replace_defaults_and_fix_uses_authored_default() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=10)
    draft.test_values = {"min_value": 15}
    assert draft.checks.values("with.checks.0")["min_value"] == 10
    assert load_yaml(draft.to_yaml())["inputs"]["min_value"]["default"] == 10
    draft.set_example(rows("17"), at=None)
    assert draft.examples[0].with_ == {"min_value": 15}
    draft.fix("with.checks.0.with.min_value")
    assert draft.test_values == {}
    assert load_yaml(draft.to_yaml())["checks"][0]["with"]["min_value"] == 10


def test_a_bound_check_reads_its_question_when_a_second_is_added() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=10)
    draft.set_questions(PAIR)
    assert draft.finalized().group
    check = load_yaml(draft.to_yaml())["checks"][0]
    assert check["question"] == "Q1"
    assert check["with"]["min_value"] == "{{ inputs.min_value }}"


def test_advanced_templates_remain_intact_when_changing_question_context() -> None:
    draft = MethodDraft.from_yaml(
        """label: Words
summary: Required words.
question_types: [text]
inputs:
  words: {default: [hello]}
checks:
  - for_each: '{{ inputs.words }}'
    uses: gradeflow/key-point@1
    with: {point: '{{ each }}'}
""",
        TEXT,
        name="words",
        version=1,
    )
    draft.set_questions(QuestionSet({**dict(TEXT), "Q2": TEXT["Q1"]}))
    assert draft.finalized().group
    assert load_yaml(draft.to_yaml())["checks"][0]["question"] == "Q1"
    draft.set_questions(TEXT)
    assert "question" not in load_yaml(draft.to_yaml())["checks"][0]
    assert draft.finalized().question_types == {"text"}


def test_every_defined_group_question_needs_an_answer_in_each_row() -> None:
    draft = numeric()
    draft.set_questions(PAIR)
    with pytest.raises(GradeFlowError, match=re.escape("Row 2 has no answer to Q2.")):
        tried(draft, [ExampleRow(answers={"Q1": "15", "Q2": "15"}), *rows("15")])


def test_saving_a_one_question_group_as_a_method_is_reusable_for_another_question() -> None:
    marking = Check.load(
        {
            "uses": "gradeflow/combination@1",
            "with": {
                "checks": [
                    {
                        "question": "Q1",
                        "uses": "gradeflow/number-range@1",
                        "with": {"min_value": 10},
                    }
                ]
            },
        }
    )
    draft = MethodDraft.from_marking(
        marking, NUMBER, name="range", version=1, label="Range", summary="At least ten."
    )
    method = draft.finalized()
    results = evaluate_rows(
        draft.ref,
        questions=QuestionSet.model_validate({"Q5": {"type": "number"}}),
        question="Q5",
        inputs={},
        rows=[ExampleRow(answers={"Q5": "12"}), ExampleRow(answers={"Q5": "8"})],
        methods=Methods([*BUILTINS, method]),
    )
    assert not method.group
    assert [row.results["Q5"].correct for row in results] == [True, False]


def test_incomplete_method_yaml_reports_plain_field_errors() -> None:
    draft = MethodDraft.from_yaml("label: []", TEXT, name="incomplete", version=1)
    assert said(draft) == ["label: Must be text."]


def test_missing_test_values_say_how_to_fix_them_naming_the_input() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Lower limit")
    draft.update_input("min_value", {"title": "Lower limit"})
    draft.test_values = {}
    assert said(draft) == ["Give “Lower limit” a test value or a default."]
    draft.test_values = {"min_value": "many"}
    assert draft.problems == [Problem("Must be a number.", field=("min_value",))]
    messages = "\n".join(said(draft))
    assert "validation error" not in messages
    assert "https://" not in messages
    assert "input_value" not in messages


def test_complex_input_defaults_are_validated_before_changing_the_draft() -> None:
    draft = MethodDraft.new(NUMBER, name="code", version=1, label="Code")
    draft.checks.add("with.checks.0", "gradeflow/custom-code@1")
    original = draft.to_yaml()
    with pytest.raises(EditingError, match="minimum: Must be an object"):
        draft.expose(
            "with.checks.0.with.parameters", label="Settings object", default={"minimum": 5}
        )
    assert draft.to_yaml() == original
    assert draft.inputs == {}
    default = {"minimum": {"type": "integer", "value": 5}}
    draft.expose("with.checks.0.with.parameters", label="Settings object", default=default)
    assert draft.test_values == {}
    assert draft.finalized().settings.model_fields["parameters"].default == default
    assert load_yaml(draft.to_yaml())["inputs"]["parameters"]["default"] == default


def test_exposed_nullable_inputs_can_have_an_explicit_none_default() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=None)
    assert draft.finalized().settings.model_fields["min_value"].default is None
    assert draft.inputs["min_value"]["default"] is None
    assert draft.test_values == {}


def test_bound_defaults_and_input_metadata_survive_unrelated_edits() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum")
    draft.update_input(
        "min_value", {"title": "Limit", "default": 4, "advanced": True, "examples": [4, 8]}
    )
    draft.checks.update("with.checks.0", {"max_value": 30, "id": "range", "name": "Range"})
    assert {field.key: field.input for field in draft.fields()}["min_value"] == "min_value"
    assert draft.checks.values("with.checks.0")["min_value"] == 4
    assert draft.problems == []
    assert "number" in draft.input_schema("min_value")["properties"]["default"]["type"]
    draft.checks.update("with.checks.0", {"min_value": 5})
    assert draft.checks.values("with.checks.0")["min_value"] == 5
    assert draft.inputs["min_value"]["examples"] == [4, 8]


def test_only_inputs_of_the_same_type_and_a_fitting_default_can_fill_a_setting() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=10)
    draft.expose("with.checks.0.with.max_value", label="Maximum", default=20)
    draft.expose("with.checks.0.name", label="Name", default="Range")
    assert draft.compatible_inputs("with.checks.0.with.min_value") == ["min_value", "max_value"]
    assert draft.compatible_inputs("with.checks.0.name") == ["name_input"]
    with pytest.raises(EditingError, match="isn't a setting an input can fill"):
        draft.compatible_inputs("with.checks.0.with.nope")


def test_an_unused_input_can_fill_any_setting_its_default_fits() -> None:
    draft = numeric()
    draft.update_input("limit", {"title": "Limit", "default": 10})
    draft.update_input("anything", {"title": "Anything"})
    assert draft.compatible_inputs("with.checks.0.with.min_value") == ["limit", "anything"]
    assert draft.compatible_inputs("with.checks.0.name") == ["anything"]
    draft.bind("with.checks.0.with.min_value", "limit")
    assert {field.key: field.input for field in draft.fields()}["min_value"] == "limit"


def test_a_new_input_for_a_setting_sharing_one_leaves_it_to_the_others() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Limit", default=10)
    draft.bind("with.checks.0.with.max_value", "min_value")
    assert draft.expose("with.checks.0.with.max_value", label="Limit") == "max_value"
    assert draft.inputs == {
        "min_value": {"title": "Limit", "default": 10},
        "max_value": {"title": "Limit 2", "default": 10},
    }
    assert [field.input for field in draft.fields() if field.input] == ["min_value", "max_value"]


def test_an_input_is_titled_as_the_setting_it_fills_unless_labelled() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value")
    [field] = [field for field in draft.fields() if field.input]
    assert draft.inputs["min_value"]["title"] == field.title
    assert field.title


def test_a_composed_check_missing_a_setting_is_a_problem_in_words() -> None:
    draft = MethodDraft.new(TEXT, name="words", version=1, label="Words")
    draft.checks.add("with.checks.0", "gradeflow/key-points@1")
    assert said(draft) == ["Words: points: Required."]


def test_whole_check_list_binding_stays_executable_in_yaml() -> None:
    draft = MethodDraft.from_yaml(
        """label: Supplied checks
summary: Supplied checks.
question_types: [number]
inputs:
  checks: {default: [{uses: gradeflow/full-marks@1}]}
checks: '{{ inputs.checks }}'
""",
        NUMBER,
        name="supplied",
        version=1,
    )
    assert draft.uneditable == ["A check list is an expression."]
    assert load_yaml(draft.to_yaml())["checks"] == "{{ inputs.checks }}"
    assert draft.problems == []


def test_a_problem_of_a_check_an_input_supplies_is_on_the_field_it_fills() -> None:
    draft = MethodDraft.from_yaml(
        """label: Supplied checks
summary: Supplied checks.
question_types: [text]
inputs:
  checks: {default: [{uses: gradeflow/full-marks@1}, {uses: gradeflow/number-range@1}]}
checks: '{{ inputs.checks }}'
""",
        TEXT,
        name="supplied",
        version=1,
    )
    [problem] = draft.checks.problems
    assert (problem.check, problem.field) == ("", ("checks", 1))


def test_metadata_edits_preserve_repeated_yaml_checks() -> None:
    draft = MethodDraft.from_yaml(
        """label: Ranges
summary: Ranges.
question_types: [number]
inputs:
  ranges: {default: [{min: 1, max: 4}]}
checks:
  - uses: gradeflow/number-range@1
    for_each: '{{ inputs.ranges }}'
    with: {min_value: '{{ each.min }}', max_value: '{{ each.max }}'}
""",
        NUMBER,
        name="ranges",
        version=1,
    )
    assert draft.problems == []
    before = load_yaml(draft.to_yaml())["checks"]
    draft.update_metadata(
        guide="Guide", level="advanced", description="Description", feedback="Feedback"
    )
    again = reopen(draft)
    assert again.metadata["level"] == "advanced"
    assert load_yaml(again.to_yaml())["checks"] == before
    assert again.uneditable == ["A check repeats for each item of an input."]


def test_nested_template_values_require_whole_document_editing() -> None:
    draft = MethodDraft.new(NUMBER, name="code", version=1, label="Code")
    draft.checks.add("with.checks.0", "gradeflow/custom-code@1")
    value = {"limit": {"type": "integer", "value": "{{ inputs.limit }}"}}
    draft.checks.update("with.checks.0", {"parameters": value})
    draft.update_input("limit", {"default": 5})
    assert draft.uneditable == ["Parameters is an expression."]
    assert reopen(draft).checks.values("with.checks.0")["parameters"] == value


def test_check_identity_edit_keeps_bound_fields_and_generated_ids_private() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=1)
    assert "id" not in draft.checks.values("with.checks.0")
    draft.checks.update("with.checks.0", {"id": "minimum", "max_value": 30})
    data = load_yaml(draft.to_yaml())
    assert data["checks"][0]["id"] == "minimum"
    assert data["checks"][0]["with"]["min_value"] == "{{ inputs.min_value }}"
    draft.checks.update("with.checks.0", {"id": "", "max_value": 40})
    data = load_yaml(draft.to_yaml())
    assert "id" not in data["checks"][0]
    assert data["checks"][0]["with"]["min_value"] == "{{ inputs.min_value }}"


def test_input_options_validate_known_default_before_mutation() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=1)
    before = draft.to_yaml()
    with pytest.raises(InvalidError, match="number"):
        draft.update_input("min_value", {"title": "Minimum", "default": {"bad": "value"}})
    assert draft.to_yaml() == before
    draft.update_input("min_value", {"title": "New minimum", "examples": [4, 8]})
    assert "default" not in draft.inputs["min_value"]


@pytest.mark.parametrize(
    "file",
    catalogue.FILES,
    ids=lambda file: file.stem,
)
def test_packaged_composed_methods_preserve_yaml_and_still_execute(file: Path) -> None:
    original = load_yaml(file.read_text())
    sample = original["examples"][0]
    questions = QuestionSet.model_validate(sample["questions"])
    draft = MethodDraft.from_yaml(
        file.read_text(),
        questions,
        name=f"gradeflow/{file.stem}",
        version=original["version"],
        test_values=sample.get("with", {}),
    )
    for example in original["examples"]:
        for row in example["rows"]:
            row.pop("expected", None)
    assert load_yaml(draft.to_yaml()) == original
    assert draft.problems == []
    # Kept whole: structures that no field holds.
    assert draft.uneditable
    assert draft.finalized().label == original["label"]
    reopened = MethodDraft.from_yaml(
        draft.to_yaml(),
        questions,
        name=draft.name,
        version=draft.version,
        test_values=draft.test_values,
    )
    assert reopened.problems == []


def test_wrapping_a_bound_check_preserves_its_binding_source() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=1)
    draft.checks.retype("with.checks.0", "gradeflow/combination@1")
    assert (
        load_yaml(draft.to_yaml())["checks"][0]["with"]["checks"][0]["with"]["min_value"]
        == "{{ inputs.min_value }}"
    )
    assert draft.problems == []
    draft.checks.update("with.checks.0.with.checks.0", {"id": "range"})
    assert draft.problems == []
    assert load_yaml(draft.to_yaml())["checks"][0]["with"]["checks"][0]["id"] == "range"


def test_json_values_with_uses_are_not_interpreted_as_nested_checks() -> None:
    draft = MethodDraft.new(NUMBER, name="code", version=1, label="Code")
    draft.checks.add("with.checks.0", "gradeflow/custom-code@1")
    draft.checks.update(
        "with.checks.0",
        {
            "parameters": {
                "item": {
                    "type": "json",
                    "value": {"uses": "gradeflow/key-point@1", "with": {"point": "cat"}},
                }
            }
        },
    )
    assert {field.check for field in draft.fields()} == {"", "with.checks.0"}
    draft.set_questions(PAIR)
    assert "question" not in draft.checks.values("with.checks.0")["parameters"]["item"]["value"]


def test_setting_one_example_keeps_the_others_and_its_title() -> None:
    draft = numeric()
    assert draft.set_example(rows("15"), at=None) == 0
    document = load_yaml(draft.to_yaml())
    document["examples"][0] |= {"title": "Ordinary", "description": "Inside range."}
    draft = MethodDraft(document, NUMBER, name="range", version=1)
    assert draft.set_example(rows("10"), at=None) == 1
    second = draft.examples[1]
    assert draft.set_example(rows("16"), at=0) == 0
    assert draft.examples[1] == second
    assert (draft.examples[0].title, draft.examples[0].description) == (
        "Ordinary",
        "Inside range.",
    )
    assert draft.examples[0].rows == rows("16")
    assert draft.set_example([], at=0) is None
    assert draft.examples == (second,)


def test_a_draft_reads_its_examples_without_results_and_keeps_those_it_is_given() -> None:
    draft = numeric()
    correct = {"Q1": ExpectedResult(marks=1, feedback="Correct.")}
    draft.set_example([ExampleRow(answers={"Q1": "15"}, expected=correct)], at=None)
    assert draft.examples[0].rows[0].expected == correct
    again = MethodDraft.from_yaml(draft.to_yaml(), NUMBER, name="range", version=1)
    assert again.examples[0].rows == rows("15")


def test_published_example_inputs_use_the_documented_yaml_alias() -> None:
    draft = numeric()
    draft.expose("with.checks.0.with.min_value", label="Minimum", default=10)
    draft.test_values = {"min_value": 12}
    draft.set_example(rows("15"), at=None)
    published = load_yaml(draft.to_yaml())
    assert published["examples"][0]["with"] == {"min_value": 12}
    assert "with_" not in published["examples"][0]


def test_examples_are_written_without_defaults_and_read_back() -> None:
    draft = numeric()
    draft.set_example([ExampleRow(answers={"Q1": "15"})], at=None)
    [example] = load_yaml(draft.to_yaml())["examples"]
    assert example["questions"] == {"Q1": {"type": "number", "max_marks": 2.0}}
    assert "with" not in example
    assert MethodDraft.from_yaml(draft.to_yaml(), draft.questions, name="range", version=1).examples


def test_an_unfinished_draft_says_what_it_is_so_far() -> None:
    text = "label: Ranges\nsummary: Within ranges.\nchecks: [{uses: missing@1}]\n"
    draft = MethodDraft.from_yaml(text, NUMBER, name="sample/ranges", version=1, methods=Methods())
    assert (draft.label, draft.summary) == ("Ranges", "Within ranges.")
    assert (draft.question_types, draft.group) == (frozenset({"number"}), False)
    group = MethodDraft.from_yaml(
        "label: 3\nchecks: []\n", PAIR, name="sample/pair", version=1, methods=Methods()
    )
    assert (group.label, group.summary, group.question_types, group.group) == (
        "",
        "",
        frozenset(),
        True,
    )


def test_the_details_form_chooses_answer_types_one_by_one() -> None:
    kinds = numeric().metadata_schema["properties"]["question_types"]
    assert kinds["type"] == "array"
    assert kinds["uniqueItems"] is True
    assert kinds["items"]["enum"] == ["text", "number", "choice", "values", "code"]
    assert kinds[HINT_KEY]["labels"]["values"] == "Multiple values"


def test_the_details_titles_tell_each_field_apart() -> None:
    titles = [prop["title"] for prop in numeric().metadata_schema["properties"].values()]
    assert len(titles) == len(set(titles))


def test_every_answer_type_is_all() -> None:
    draft = numeric()
    draft.update_metadata(question_types=["number", "text", "choice", "values", "code"])
    assert load_yaml(draft.to_yaml())["question_types"] == "all"
    assert draft.metadata["question_types"] == ["text", "number", "choice", "values", "code"]


def test_metadata_saved_as_it_is_leaves_the_file_unchanged() -> None:
    draft = numeric()
    draft.update_metadata(
        question_types=["text", "number", "choice", "values", "code"], level="advanced"
    )
    before = draft.to_yaml()
    draft.update_metadata(**draft.metadata)
    assert draft.to_yaml() == before


def test_defaults_and_empty_optional_metadata_are_left_unset() -> None:
    draft = numeric()
    before = load_yaml(draft.to_yaml())
    draft.update_metadata(guide="", level="basic", description="", feedback="")
    assert load_yaml(draft.to_yaml()) == before


def test_without_answer_types_a_draft_marks_its_questions_type() -> None:
    draft = numeric()
    draft.update_metadata(question_types=["number", "text"])
    draft.update_metadata(question_types=[])
    assert draft.question_types == {"number"}


def test_metadata_set_to_none_is_unset() -> None:
    draft = numeric()
    draft.update_metadata(feedback="Well done.")
    draft.update_metadata(feedback=None)
    assert "feedback" not in draft.metadata
    assert "feedback" not in load_yaml(draft.to_yaml())


def test_a_name_an_input_fills_labels_its_check_with_its_default() -> None:
    draft = numeric()
    draft.checks.update("with.checks.0", {"name": "In range"})
    draft.expose("with.checks.0.name", label="Check name")
    assert draft.checks.label("with.checks.0") == "In range"
    draft.update_input("name_input", {"title": "Check name"})
    assert draft.checks.label("with.checks.0") == "Number Range"


def test_a_setting_an_input_fills_has_its_problems_with_the_method() -> None:
    draft = MethodDraft.new(TEXT, name="answers", version=1, label="Answers")
    draft.checks.add("with.checks.0", "gradeflow/accepted-answers@1")
    draft.expose("with.checks.0.with.answers", label="Answers")
    assert draft.checks.problems == []
    assert draft.problems
    with pytest.raises(EditingError):
        draft.checks.finalized()


def test_a_change_to_a_check_thats_not_there_is_refused() -> None:
    draft = numeric()
    with pytest.raises(EditingError):
        draft.checks.update("with.checks.5", {"min_value": 1})
