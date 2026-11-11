import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from gradeflow_core import (
    BUILTINS,
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
    MethodDraft,
    Position,
    RowResult,
    load_yaml,
    mark_rows,
)
from gradeflow_core.methods import catalogue
from gradeflow_core.steps.step import valid_step

from .helpers import NUMBER_QUESTIONS, number_range_draft

TWO_NUMBER_QUESTIONS = QuestionSet.load({"Q1": {"type": "number"}, "Q2": {"type": "number"}})
TEXT_QUESTIONS = QuestionSet.load({"Q1": {"type": "text"}})


def reopen(draft: MethodDraft, *, version: int = 1) -> MethodDraft:
    return MethodDraft.from_yaml(draft.to_yaml(), draft.questions, name=draft.name, version=version)


def problem_messages(draft: MethodDraft) -> list[str]:
    return [str(problem) for problem in draft.problems]


def example_rows(*answers: str) -> list[ExampleRow]:
    return [ExampleRow(answers={"Q1": answer}) for answer in answers]


def try_rows(
    draft: MethodDraft, examples: list[ExampleRow], methods: Methods = BUILTINS
) -> list[RowResult]:
    method = draft.finalized()
    return mark_rows(
        method.ref,
        questions=draft.questions,
        question=draft.question,
        settings=draft.test_values,
        rows=examples,
        methods=Methods([*methods, method]),
    )


def test_a_new_draft_is_an_empty_workflow_written_with_its_version_first() -> None:
    draft = MethodDraft.new(NUMBER_QUESTIONS, name="empty", version=3, label="Empty")
    assert draft.steps_draft.data == {
        "question": "Q1",
        "uses": "gradeflow/workflow@1",
        "with": {"steps": []},
    }
    assert list(load_yaml(draft.to_yaml())) == [
        "version",
        "label",
        "summary",
        "question_types",
        "steps",
    ]
    assert load_yaml(draft.to_yaml())["version"] == 3


def test_a_files_steps_are_edited_as_a_workflow_whose_method_is_fixed() -> None:
    reopened = reopen(number_range_draft())
    assert reopened.steps_draft.data == number_range_draft().steps_draft.data
    assert reopened.to_yaml() == number_range_draft().to_yaml()
    assert [method.ref for method in reopened.steps_draft.methods("")] == ["gradeflow/workflow@1"]
    assert (
        not {"name", "id", "if", "feedback"} & reopened.steps_draft.schema("")["properties"].keys()
    )

    reopened.steps_draft.add("/with/steps/1", "gradeflow/full-marks@1")
    assert reopen(reopened).steps_draft.is_step("/with/steps/1")

    proportional_draft = number_range_draft()
    proportional_draft.steps_draft.update("", {"marks_by": "proportional"})
    assert reopen(proportional_draft).steps_draft.data["with"]["marks_by"] == "proportional"


def test_bound_settings_follow_moved_checks_and_examples_never_change_defaults() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.bind_new_setting("/with/steps/0/with/max_value")
    draft.steps_draft.add("/with/steps/1", "gradeflow/full-marks@1")
    draft.steps_draft.move("/with/steps/0", 1)
    text = draft.to_yaml()
    document = load_yaml(text)
    assert document["version"] == 1
    assert document["steps"][1]["with"]["min_value"] == "{{ settings.min_value }}"
    assert document["settings"]["min_value"]["default"] == 10
    assert document["settings"]["max_value"]["default"] == 20
    reopened = MethodDraft.from_yaml(
        text, draft.questions, name="range", version=1, test_values={"max_value": 25}
    )
    assert reopened.steps_draft.values("/with/steps/1")["max_value"] == 20
    assert load_yaml(reopened.to_yaml())["settings"]["max_value"]["default"] == 20
    assert reopened.settings_schema["properties"]["min_value"]["title"] == "From"
    version_4 = reopen(reopened, version=4)
    assert BUILTINS.with_files([("sample/range", version_4.to_yaml())])[-1].ref == "sample/range@4"


def test_deletion_and_retyping_report_affected_settings_and_allow_explicit_removal() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.steps_draft.remove("/with/steps/0")
    assert "From isn't used." in problem_messages(draft)
    with pytest.raises(EditingError, match="From"):
        draft.finalized()
    draft.remove_declared_setting("min_value")
    assert "settings" not in load_yaml(draft.to_yaml())


def test_repeated_yaml_stays_executable_and_requires_whole_document_editing() -> None:
    text = """label: Repeated
summary: Repeated checks.
question_types: [text]
settings:
  terms: {default: [a, b]}
steps:
  - for_each: '{{ settings.terms }}'
    uses: gradeflow/key-point@1
    with: {point: '{{ item }}'}
"""
    draft = MethodDraft.from_yaml(text, TEXT_QUESTIONS, name="repeated", version=1)
    assert not [problem for problem in problem_messages(draft) if "isn't used" in problem]
    assert load_yaml(draft.to_yaml()) == {"version": 1, **load_yaml(text)}
    assert draft.uneditable_reasons == ["A step repeats for each item of a setting."]
    assert [row.results["Q1"].correct for row in try_rows(draft, example_rows("a b", "c"))] == [
        True,
        False,
    ]
    draft.steps_draft.update("/with/steps/0", {"name": "Term"})
    assert list(draft.steps_draft.data["with"]["steps"][0].items()) == [
        ("for_each", "{{ settings.terms }}"),
        ("name", "Term"),
        ("uses", "gradeflow/key-point@1"),
        ("with", {"point": "{{ item }}"}),
    ]
    assert draft.problems == []


def test_compatibility_refuses_a_number_check_declared_for_text() -> None:
    draft = number_range_draft()
    draft.update_metadata(question_types=["number", "text"])
    assert any(
        problem.startswith("It can't mark text questions: ") for problem in problem_messages(draft)
    )
    with pytest.raises(EditingError, match="It can't mark text questions"):
        draft.finalized()


def test_template_only_settings_are_used_even_without_bound_fields() -> None:
    text = """label: Conditional
summary: Optional check.
question_types: [number]
settings:
  enabled: {default: true}
  message: {default: Good.}
  unused: {default: 10}
steps:
  - uses: gradeflow/full-marks@1
    if: '{{ settings.enabled }}'
feedback: '{{ settings.message }}'
"""
    draft = MethodDraft.from_yaml(text, NUMBER_QUESTIONS, name="conditional", version=1)
    assert [problem for problem in problem_messages(draft) if "isn't used" in problem] == [
        "Setting unused isn't used."
    ]


def test_trying_rows_leaves_the_examples_unchanged() -> None:
    draft = number_range_draft()
    expected = {"Q1": {"marks": 2, "feedback": "Correct."}}
    results = try_rows(
        draft,
        [
            ExampleRow(answers={"Q1": "15"}),
            ExampleRow.model_validate({"answers": {"Q1": "8"}, "expected": expected}),
            ExampleRow(answers={"Q1": "invalid"}),
            ExampleRow(answers={"Q1": ""}),
        ],
    )
    assert (results[0].total_marks, results[0].matches_expected) == (2, None)
    assert (results[1].total_marks, results[1].matches_expected) == (0, False)
    assert not results[2].results["Q1"].is_marked
    assert results[3].total_marks == 0
    assert draft.examples == ()


@pytest.mark.parametrize("version", [0, -1])
def test_a_version_is_positive(version: int) -> None:
    text = number_range_draft().to_yaml()
    draft = MethodDraft.from_yaml(text, NUMBER_QUESTIONS, name="range", version=version)
    with pytest.raises(EditingError, match="Must be more than 0"):
        draft.finalized()


def test_versions_are_explicit_and_written_by_the_draft() -> None:
    text = number_range_draft().to_yaml()
    with pytest.raises(MethodFileError, match="version"):
        BUILTINS.with_files([("sample/range", text.replace("version: 1\n", ""))])
    version_5 = MethodDraft.from_yaml(text, NUMBER_QUESTIONS, name="range", version=5)
    assert version_5.to_yaml() == text.replace("version: 1", "version: 5")
    assert (
        MethodDraft.from_yaml(
            version_5.to_yaml(), NUMBER_QUESTIONS, name="range", version=1
        ).to_yaml()
        == text
    )
    assert all(method.version >= 1 for method in BUILTINS)


def test_group_question_settings_are_reusable_and_all_sample_answers_are_required() -> None:
    draft = MethodDraft.new(
        TWO_NUMBER_QUESTIONS, name="pair", version=1, label="Pair", summary="Both."
    )
    draft.steps_draft.add("/with/steps/0", "gradeflow/number-range@1", "Q1")
    draft.steps_draft.add("/with/steps/1", "gradeflow/number-range@1", "Q2")
    draft.steps_draft.update("/with/steps/0", {"min_value": 10})
    draft.steps_draft.update("/with/steps/1", {"max_value": 20})
    draft.bind_new_setting("/with/steps/0/question")
    draft.bind_new_setting("/with/steps/1/question")
    [result] = try_rows(draft, [ExampleRow(answers={"Q1": "15", "Q2": "25"})])
    assert result.total_marks == 1
    assert result.results["Q1"].correct
    assert not result.results["Q2"].correct
    with pytest.raises(GradeFlowError, match=re.escape("Row 1 has no answer to Q2.")):
        try_rows(draft, example_rows("15"))


def test_retyping_a_bound_check_refuses_to_rebind_it_to_the_new_method() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.steps_draft.change_method("/with/steps/0", "gradeflow/full-marks@1")
    assert problem_messages(draft) == ["From isn't used."]
    assert draft.declared_settings["min_value"]["default"] == 10
    draft.remove_declared_setting("min_value")
    assert draft.problems == []


def test_a_method_whose_root_gives_a_step_output_is_a_transform_and_keeps_its_steps() -> None:
    draft = MethodDraft.new(TEXT_QUESTIONS, name="last", version=1, label="Last", summary="Last.")
    draft.steps_draft.add("/with/steps/0", "gradeflow/extract@1")
    draft.steps_draft.update("/with/steps/0", {"occurrence": "last"})
    steps = draft.steps_draft.data["with"]["steps"]
    output_step = f"{{{{ steps.{steps[0]['id']}.output }}}}"
    draft.steps_draft.update("", {"output": "step_output", "output_step": output_step})
    assert draft.steps_draft.data["with"]["steps"] == steps
    document = load_yaml(draft.to_yaml())
    assert (document["output"], document["output_step"]) == ("step_output", output_step)
    assert not reopen(draft).finalized().gives_marks({})
    [result] = try_rows(reopen(draft), example_rows("7 then 11"))
    assert result.results["Q1"].feedback == "Read as: 11"


def test_from_marking_preserves_templates_written_as_literal_answer_text() -> None:
    check = valid_step(
        {"uses": "gradeflow/key-point@1", "question": "Q1", "with": {"point": "{{ literal }}"}},
        BUILTINS.by_ref,
    )
    draft = MethodDraft.from_marking(
        check, TEXT_QUESTIONS, name="braces", version=1, label="Braces", summary="Braces."
    )
    [result] = try_rows(draft, example_rows("{{ literal }}"))
    assert result.total_marks == 1


def test_from_marking_keeps_braces_in_check_names_but_not_in_feedback() -> None:
    check = valid_step(
        {
            "question": "Q1",
            "uses": "gradeflow/workflow@1",
            "with": {
                "steps": [
                    {
                        "uses": "gradeflow/key-point@1",
                        "name": "Has {{ braces }}",
                        "with": {"point": "a"},
                        "feedback": "{{ feedback }} Got {{ answer }}.",
                    },
                    {"uses": "gradeflow/key-point@1", "with": {"point": "b"}},
                ]
            },
        },
        BUILTINS.by_ref,
    )
    draft = MethodDraft.from_marking(
        check, TEXT_QUESTIONS, name="braces", version=1, label="Braces", summary="Braces."
    )
    [result] = try_rows(draft, example_rows("b"))
    assert result.results["Q1"].feedback.startswith("Has {{ braces }}: Missing “a”. Got b.")


def test_invalid_setting_defaults_are_rejected_even_when_test_values_are_valid() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    document = load_yaml(draft.to_yaml())
    document["settings"]["min_value"]["default"] = "not a number"
    reopened = MethodDraft(
        document, NUMBER_QUESTIONS, name="range", version=1, test_values={"min_value": 10}
    )
    with pytest.raises(EditingError, match="number"):
        reopened.finalized()


def test_unbinding_a_bound_setting_preserves_its_inherited_default() -> None:
    draft = MethodDraft.new(TEXT_QUESTIONS, name="cat", version=1, label="Cat")
    draft.steps_draft.add("/with/steps/0", "gradeflow/key-points@1")
    draft.steps_draft.update("/with/steps/0", {"points": ["cat"]})
    draft.bind_new_setting("/with/steps/0/with/whole_words")
    reopened = reopen(draft)
    assert reopened.steps_draft.values("/with/steps/0")["whole_words"] is False
    reopened.unbind("/with/steps/0/with/whole_words")
    assert reopened.problems == []


def test_a_refused_change_to_a_bound_field_leaves_its_default() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft_using_removed_method = MethodDraft.from_yaml(
        draft.to_yaml().replace("gradeflow/number-range@1", "gradeflow/retired@1"),
        NUMBER_QUESTIONS,
        name="range",
        version=1,
    )
    with pytest.raises(EditingError, match=re.escape("There's no method gradeflow/retired@1.")):
        draft_using_removed_method.steps_draft.update("/with/steps/0", {"min_value": 3})
    assert draft_using_removed_method.declared_settings["min_value"]["default"] == 10


def test_removing_a_setting_without_a_default_leaves_the_field_unset() -> None:
    draft = number_range_draft()
    name = draft.bind_new_setting("/with/steps/0/weight")
    draft.update_declared_setting(name, {"title": "Weight"})
    draft.remove_declared_setting(name)
    assert "weight" not in draft.steps_draft.data["with"]["steps"][0]
    assert draft.problems == []


def test_opening_a_bound_definition_does_not_change_publishable_content() -> None:
    draft = reopen(number_range_draft(), version=3)
    draft.bind_new_setting("/with/steps/0/with/min_value")
    text = draft.to_yaml()
    assert reopen(draft, version=3).to_yaml() == text


def test_a_new_version_may_use_the_previous_one_but_not_itself() -> None:
    published = number_range_draft().finalized()
    text = number_range_draft().to_yaml()
    edited = MethodDraft.from_yaml(
        text, NUMBER_QUESTIONS, name="range", version=2, methods=Methods([*BUILTINS, published])
    )
    edited.steps_draft.update("/with/steps/0", {"max_value": 25})
    assert edited.finalized().ref == "range@2"
    assert published.version == 1
    edited.steps_draft.change_method("/with/steps/0", "range@1")
    assert edited.problems == []
    assert [
        row.total_marks
        for row in try_rows(edited, example_rows("15", "25"), Methods([*BUILTINS, published]))
    ] == [2, 0]
    same_version = MethodDraft.from_yaml(
        text, NUMBER_QUESTIONS, name="range", version=1, methods=Methods([*BUILTINS, published])
    )
    same_version.steps_draft.change_method("/with/steps/0", "range@1")
    with pytest.raises(EditingError, match=re.escape("There's no method range@1.")):
        same_version.finalized()


def test_bound_setting_keeps_its_condition_reference_after_reload() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.steps_draft.add("/with/steps/1", "gradeflow/full-marks@1")
    draft.steps_draft.update(
        "/with/steps/1", {"if": Condition((ConditionRow("/with/steps/0", "correct"),))}
    )
    reopened = reopen(draft)
    assert reopened.problems == []
    assert isinstance(reopened.steps_draft.values("/with/steps/1")["if"], Condition)
    field = next(
        field for field in reopened.configurable_fields() if field.bound_setting == "min_value"
    )
    assert (field.step_path, field.key) == ("/with/steps/0", "min_value")


@pytest.mark.parametrize(
    "content",
    [
        "steps: 2",
        "steps: [{uses: gradeflow/full-marks@1, with: bad}]",
    ],
)
def test_incomplete_yaml_remains_a_draft_with_problems(content: str) -> None:
    text = f"label: Incomplete\nsummary: Incomplete.\nquestion_types: [text]\n{content}\n"
    draft = MethodDraft.from_yaml(text, TEXT_QUESTIONS, name="incomplete", version=1)
    assert draft.to_yaml()
    assert draft.problems


def test_settings_that_arent_declarations_cant_be_opened() -> None:
    text = "label: Incomplete\nsummary: Incomplete.\nsettings: {a: null}\nsteps: []\n"
    with pytest.raises(
        EditingError, match=re.escape("Settings must map each name to its declaration.")
    ):
        MethodDraft.from_yaml(text, TEXT_QUESTIONS, name="incomplete", version=1)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("- label: Listed\n", Problem("A method file must be a set of keys and values.")),
        ("label: [\n", Problem("expected the node content, but found '<stream end>'.", line=2)),
    ],
)
def test_text_that_isnt_a_method_file_cant_be_opened(text: str, problem: Problem) -> None:
    with pytest.raises(InvalidError) as exc_info:
        MethodDraft.from_yaml(text, TEXT_QUESTIONS, name="incomplete", version=1)
    assert exc_info.value.problems == [problem]


def test_two_questions_make_a_group_method_and_keep_bound_checks() -> None:
    draft = MethodDraft.new(NUMBER_QUESTIONS, name="range", version=1, label="Range")
    draft.steps_draft.add("/with/steps/0", "gradeflow/number-range@1")
    draft.steps_draft.update("/with/steps/0", {"min_value": 10})
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.set_questions(TWO_NUMBER_QUESTIONS)
    assert (draft.question, draft.finalized().marks_question_group) == (None, True)
    assert draft.metadata["marks_question_group"] is True
    draft.update_metadata(**draft.metadata)
    assert draft.steps_draft.values("/with/steps/0")["question"] == "Q1"
    assert (
        next(
            field for field in draft.configurable_fields() if field.key == "min_value"
        ).bound_setting
        == "min_value"
    )
    assert draft.test_values == {}
    draft.set_questions(NUMBER_QUESTIONS)
    assert draft.question == "Q1"
    assert not draft.finalized().marks_question_group
    assert "question" not in load_yaml(draft.to_yaml())["steps"][0]
    assert draft.test_values == {}


def test_checks_of_removed_questions_are_reported_and_drafts_without_questions_cannot_publish() -> (
    None
):
    draft = MethodDraft.new(TWO_NUMBER_QUESTIONS, name="range", version=1, label="Range")
    draft.steps_draft.add("/with/steps/0", "gradeflow/number-range@1", "Q2")
    draft.steps_draft.update("/with/steps/0", {"min_value": 10})
    draft.set_questions(QuestionSet({"Q1": TWO_NUMBER_QUESTIONS["Q1"]}))
    assert draft.steps_draft.values("/with/steps/0")["question"] == "Q2"
    assert "Q2" in " ".join(problem_messages(draft))
    draft.set_questions(QuestionSet({}))
    assert problem_messages(draft) == ["Add at least one question."]
    with pytest.raises(EditingError, match=re.escape("Add at least one question.")):
        draft.finalized()


def test_a_newly_bound_setting_uses_its_current_value_as_the_default() -> None:
    draft = MethodDraft.new(TEXT_QUESTIONS, name="cat", version=1, label="Cat")
    draft.steps_draft.add("/with/steps/0", "gradeflow/key-points@1")
    draft.steps_draft.update("/with/steps/0", {"points": ["cat"]})
    draft.bind_new_setting("/with/steps/0/with/whole_words")
    assert draft.test_values == {}
    assert "whole_words" not in draft.settings_schema.get("required", [])
    schema = draft.finalized().settings_schema(Position(TEXT_QUESTIONS))
    assert schema["properties"]["whole_words"]["default"] is False
    draft.unbind("/with/steps/0/with/whole_words")
    assert draft.test_values == {}
    assert not draft.finalized().settings_schema(Position(TEXT_QUESTIONS))["properties"]


def test_examples_publish_without_execution_or_fabricated_expectations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = number_range_draft()
    mark = Mock()
    monkeypatch.setattr("gradeflow_core.marking.examples.mark", mark)
    wrong_row = ExampleRow.model_validate(
        {"answers": {"Q1": "8"}, "expected": {"Q1": {"marks": 2, "feedback": "Wrong."}}}
    )
    index = draft.set_example(
        [
            ExampleRow(answers={"Q1": "15"}),
            wrong_row,
        ],
        index=None,
    )
    assert index == 0
    method = draft.finalized()
    first, second = method.examples[0].rows
    assert (first.answers, first.expected) == ({"Q1": "15"}, {})
    assert second.expected == wrong_row.expected
    assert method.examples[0].questions == dict(draft.questions)
    assert draft.set_example([], index=0) is None
    assert draft.finalized().examples == ()
    mark.assert_not_called()


def test_the_tested_step_is_the_method_with_its_test_values() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.test_values = {"min_value": 15}
    step = draft.tested_step()
    assert step.dump() == {"question": "Q1", "uses": "range@1", "with": {"min_value": 15}}
    assert (step.questions_read(), step.needs_model_endpoint()) == ({"Q1"}, False)
    draft.steps_draft.update("/with/steps/0", {"max_value": 5})
    with pytest.raises(EditingError):
        draft.tested_step()


def test_changed_test_values_do_not_replace_defaults_and_fix_uses_authored_default() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.test_values = {"min_value": 15}
    assert draft.steps_draft.values("/with/steps/0")["min_value"] == 10
    assert load_yaml(draft.to_yaml())["settings"]["min_value"]["default"] == 10
    draft.set_example(example_rows("17"), index=None)
    assert draft.examples[0].with_ == {"min_value": 15}
    draft.unbind("/with/steps/0/with/min_value")
    assert draft.test_values == {}
    assert load_yaml(draft.to_yaml())["steps"][0]["with"]["min_value"] == 10


def test_a_bound_check_reads_its_question_when_a_second_is_added() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.set_questions(TWO_NUMBER_QUESTIONS)
    assert draft.finalized().marks_question_group
    check = load_yaml(draft.to_yaml())["steps"][0]
    assert check["question"] == "Q1"
    assert check["with"]["min_value"] == "{{ settings.min_value }}"


def test_advanced_templates_remain_intact_when_changing_question_context() -> None:
    draft = MethodDraft.from_yaml(
        """label: Words
summary: Required words.
question_types: [text]
settings:
  words: {default: [hello]}
steps:
  - for_each: '{{ settings.words }}'
    uses: gradeflow/key-point@1
    with: {point: '{{ item }}'}
""",
        TEXT_QUESTIONS,
        name="words",
        version=1,
    )
    draft.set_questions(QuestionSet({**dict(TEXT_QUESTIONS), "Q2": TEXT_QUESTIONS["Q1"]}))
    assert draft.finalized().marks_question_group
    assert load_yaml(draft.to_yaml())["steps"][0]["question"] == "Q1"
    draft.set_questions(TEXT_QUESTIONS)
    assert "question" not in load_yaml(draft.to_yaml())["steps"][0]
    assert draft.finalized().question_types == {"text"}


def test_every_defined_group_question_needs_an_answer_in_each_row() -> None:
    draft = number_range_draft()
    draft.set_questions(TWO_NUMBER_QUESTIONS)
    with pytest.raises(GradeFlowError, match=re.escape("Row 2 has no answer to Q2.")):
        try_rows(draft, [ExampleRow(answers={"Q1": "15", "Q2": "15"}), *example_rows("15")])


def test_saving_a_one_question_group_as_a_method_is_reusable_for_another_question() -> None:
    marking = valid_step(
        {
            "uses": "gradeflow/workflow@1",
            "with": {
                "steps": [
                    {
                        "question": "Q1",
                        "uses": "gradeflow/number-range@1",
                        "with": {"min_value": 10},
                    }
                ]
            },
        },
        BUILTINS.by_ref,
    )
    draft = MethodDraft.from_marking(
        marking, NUMBER_QUESTIONS, name="range", version=1, label="Range", summary="At least ten."
    )
    method = draft.finalized()
    results = mark_rows(
        method.ref,
        questions=QuestionSet.load({"Q5": {"type": "number"}}),
        question="Q5",
        settings={},
        rows=[ExampleRow(answers={"Q5": "12"}), ExampleRow(answers={"Q5": "8"})],
        methods=Methods([*BUILTINS, method]),
    )
    assert not method.marks_question_group
    assert [row.results["Q5"].correct for row in results] == [True, False]


def test_incomplete_method_yaml_reports_plain_field_errors() -> None:
    draft = MethodDraft.from_yaml("label: []", TEXT_QUESTIONS, name="incomplete", version=1)
    assert problem_messages(draft) == ["label: Must be text."]


def test_missing_test_values_say_how_to_fix_them_naming_the_setting() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.update_declared_setting("min_value", {"title": "Lower limit"})
    draft.test_values = {}
    assert problem_messages(draft) == ["Give “Lower limit” a test value or a default."]
    draft.test_values = {"min_value": "many"}
    assert draft.problems == [Problem("Must be a number.", field=("min_value",))]
    messages = "\n".join(problem_messages(draft))
    assert "validation error" not in messages
    assert "https://" not in messages
    assert "input_value" not in messages


def test_a_complex_field_value_becomes_its_settings_default() -> None:
    draft = MethodDraft.new(NUMBER_QUESTIONS, name="code", version=1, label="Code")
    draft.steps_draft.add("/with/steps/0", "gradeflow/custom-code@1")
    default = {"minimum": {"type": "integer", "value": 5}}
    draft.steps_draft.update("/with/steps/0", {"parameters": default})
    draft.bind_new_setting("/with/steps/0/with/parameters")
    assert draft.test_values == {}
    schema = draft.finalized().settings_schema(Position(NUMBER_QUESTIONS))
    assert schema["properties"]["parameters"]["default"] == default
    assert load_yaml(draft.to_yaml())["settings"]["parameters"]["default"] == default


def test_bound_defaults_and_setting_metadata_survive_unrelated_edits() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.update_declared_setting(
        "min_value", {"title": "Limit", "default": 4, "advanced": True, "examples": [4, 8]}
    )
    draft.steps_draft.update("/with/steps/0", {"max_value": 30, "id": "range", "name": "Range"})
    assert {field.key: field.bound_setting for field in draft.configurable_fields()}[
        "min_value"
    ] == "min_value"
    assert draft.steps_draft.values("/with/steps/0")["min_value"] == 4
    assert draft.problems == []
    assert "number" in draft.declared_setting_schema("min_value")["properties"]["default"]["type"]
    draft.steps_draft.update("/with/steps/0", {"min_value": 5})
    assert draft.steps_draft.values("/with/steps/0")["min_value"] == 5
    assert draft.declared_settings["min_value"]["examples"] == [4, 8]


def test_only_settings_of_the_same_type_and_a_fitting_default_can_fill_a_field() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.bind_new_setting("/with/steps/0/with/max_value")
    draft.steps_draft.update("/with/steps/0", {"name": "Range"})
    draft.bind_new_setting("/with/steps/0/name")
    assert draft.compatible_settings("/with/steps/0/with/min_value") == ["min_value", "max_value"]
    assert draft.compatible_settings("/with/steps/0/name") == ["name_setting"]
    with pytest.raises(EditingError, match="isn't a field a setting can fill"):
        draft.compatible_settings("/with/steps/0/with/nope")


def test_an_unused_setting_can_fill_any_field_its_default_fits() -> None:
    draft = number_range_draft()
    draft.update_declared_setting("limit", {"title": "Limit", "default": 10})
    draft.update_declared_setting("anything", {"title": "Anything"})
    assert draft.compatible_settings("/with/steps/0/with/min_value") == ["limit", "anything"]
    assert draft.compatible_settings("/with/steps/0/name") == ["anything"]
    draft.bind("/with/steps/0/with/min_value", "limit")
    assert {field.key: field.bound_setting for field in draft.configurable_fields()}[
        "min_value"
    ] == "limit"


def test_binding_a_new_setting_to_a_field_sharing_one_leaves_the_other_fields_bound() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.update_declared_setting("min_value", {"title": "To", "default": 10})
    draft.bind("/with/steps/0/with/max_value", "min_value")
    assert draft.bind_new_setting("/with/steps/0/with/max_value") == "max_value"
    assert draft.declared_settings == {
        "min_value": {"title": "To", "default": 10},
        "max_value": {"title": "To 2", "default": 10},
    }
    assert [
        field.bound_setting for field in draft.configurable_fields() if field.bound_setting
    ] == [
        "min_value",
        "max_value",
    ]


def test_a_setting_is_titled_as_the_field_it_fills() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    [field] = [field for field in draft.configurable_fields() if field.bound_setting]
    assert draft.declared_settings["min_value"]["title"] == field.title
    assert field.title


def test_a_composed_check_missing_a_setting_is_a_problem_in_words() -> None:
    draft = MethodDraft.new(TEXT_QUESTIONS, name="words", version=1, label="Words")
    draft.steps_draft.add("/with/steps/0", "gradeflow/key-points@1")
    assert draft.problems == [
        Problem("Required.", "/with/steps/0", ("points",), is_incomplete=True)
    ]


def test_a_method_that_doesnt_load_lists_each_problem() -> None:
    draft = MethodDraft.from_yaml(
        """label: Old
summary: Written for an earlier version.
question_types: [text]
steps:
  - uses: gradeflow/key-points@1
    with: {points: [light], old: 1}
  - uses: gradeflow/best-answer@1
""",
        TEXT_QUESTIONS,
        name="sample/old",
        version=1,
    )
    assert problem_messages(draft) == [
        "Key Points has no setting old.",
        "There's no method gradeflow/best-answer@1.",
    ]


def test_whole_check_list_binding_stays_executable_in_yaml() -> None:
    draft = MethodDraft.from_yaml(
        """label: Supplied checks
summary: Supplied checks.
question_types: [number]
settings:
  checks: {default: [{uses: gradeflow/full-marks@1}]}
steps: '{{ settings.checks }}'
""",
        NUMBER_QUESTIONS,
        name="supplied",
        version=1,
    )
    assert draft.uneditable_reasons == ["A list of steps is written as a template."]
    assert load_yaml(draft.to_yaml())["steps"] == "{{ settings.checks }}"
    assert draft.problems == []


def test_a_problem_of_a_check_a_setting_supplies_is_on_the_field_it_fills() -> None:
    draft = MethodDraft.from_yaml(
        """label: Supplied checks
summary: Supplied checks.
question_types: [text]
settings:
  checks: {default: [{uses: gradeflow/full-marks@1}, {uses: gradeflow/number-range@1}]}
steps: '{{ settings.checks }}'
""",
        TEXT_QUESTIONS,
        name="supplied",
        version=1,
    )
    [problem] = draft.steps_draft.problems
    assert (problem.step_path, problem.field) == ("", ("steps", 1))


def test_metadata_edits_preserve_repeated_yaml_checks() -> None:
    draft = MethodDraft.from_yaml(
        """label: Ranges
summary: Ranges.
question_types: [number]
settings:
  ranges: {default: [{min: 1, max: 4}]}
steps:
  - uses: gradeflow/number-range@1
    for_each: '{{ settings.ranges }}'
    with: {min_value: '{{ item.min }}', max_value: '{{ item.max }}'}
""",
        NUMBER_QUESTIONS,
        name="ranges",
        version=1,
    )
    assert draft.problems == []
    before = load_yaml(draft.to_yaml())["steps"]
    draft.update_metadata(
        guide="Guide", level="advanced", description="Description", feedback="Feedback"
    )
    reopened = reopen(draft)
    assert reopened.metadata["level"] == "advanced"
    assert load_yaml(reopened.to_yaml())["steps"] == before
    assert reopened.uneditable_reasons == ["A step repeats for each item of a setting."]


def test_nested_template_values_require_whole_document_editing() -> None:
    draft = MethodDraft.new(NUMBER_QUESTIONS, name="code", version=1, label="Code")
    draft.steps_draft.add("/with/steps/0", "gradeflow/custom-code@1")
    parameters = {"limit": {"type": "integer", "value": "{{ settings.limit }}"}}
    draft.steps_draft.update("/with/steps/0", {"parameters": parameters})
    draft.update_declared_setting("limit", {"default": 5})
    assert draft.uneditable_reasons == ["Parameters is written as a template."]
    assert reopen(draft).steps_draft.values("/with/steps/0")["parameters"] == parameters


def test_editing_a_checks_id_keeps_bound_fields_and_hides_generated_ids() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    assert "id" not in draft.steps_draft.values("/with/steps/0")
    draft.steps_draft.update("/with/steps/0", {"id": "minimum", "max_value": 30})
    document = load_yaml(draft.to_yaml())
    assert document["steps"][0]["id"] == "minimum"
    assert document["steps"][0]["with"]["min_value"] == "{{ settings.min_value }}"
    draft.steps_draft.update("/with/steps/0", {"id": "", "max_value": 40})
    document = load_yaml(draft.to_yaml())
    assert "id" not in document["steps"][0]
    assert document["steps"][0]["with"]["min_value"] == "{{ settings.min_value }}"


def test_setting_options_validate_known_default_before_mutation() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    before = draft.to_yaml()
    with pytest.raises(InvalidError, match="number"):
        draft.update_declared_setting(
            "min_value", {"title": "Minimum", "default": {"bad": "value"}}
        )
    assert draft.to_yaml() == before
    draft.update_declared_setting("min_value", {"title": "New minimum", "examples": [4, 8]})
    assert "default" not in draft.declared_settings["min_value"]


@pytest.mark.parametrize(
    "file",
    catalogue.METHOD_FILE_PATHS,
    ids=lambda file: file.stem,
)
def test_packaged_composed_methods_preserve_yaml_and_still_execute(file: Path) -> None:
    original = load_yaml(file.read_text())
    first_example = original["examples"][0]
    questions = QuestionSet.load(first_example["questions"])
    draft = MethodDraft.from_yaml(
        file.read_text(),
        questions,
        name=f"gradeflow/{file.stem}",
        version=original["version"],
        test_values=first_example.get("with", {}),
    )
    for example in original["examples"]:
        for row in example["rows"]:
            row.pop("expected", None)
    assert load_yaml(draft.to_yaml()) == original
    assert draft.problems == []
    # Kept whole: structures that no field holds.
    assert draft.uneditable_reasons
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
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.steps_draft.change_method("/with/steps/0", "gradeflow/workflow@1")
    assert (
        load_yaml(draft.to_yaml())["steps"][0]["with"]["steps"][0]["with"]["min_value"]
        == "{{ settings.min_value }}"
    )
    assert draft.problems == []
    draft.steps_draft.update("/with/steps/0/with/steps/0", {"id": "range"})
    assert draft.problems == []
    assert load_yaml(draft.to_yaml())["steps"][0]["with"]["steps"][0]["id"] == "range"


def test_json_values_with_uses_are_not_interpreted_as_nested_checks() -> None:
    draft = MethodDraft.new(NUMBER_QUESTIONS, name="code", version=1, label="Code")
    draft.steps_draft.add("/with/steps/0", "gradeflow/custom-code@1")
    draft.steps_draft.update(
        "/with/steps/0",
        {
            "parameters": {
                "item": {
                    "type": "json",
                    "value": {"uses": "gradeflow/key-point@1", "with": {"point": "cat"}},
                }
            }
        },
    )
    assert {field.step_path for field in draft.configurable_fields()} == {"", "/with/steps/0"}
    draft.set_questions(TWO_NUMBER_QUESTIONS)
    assert (
        "question" not in draft.steps_draft.values("/with/steps/0")["parameters"]["item"]["value"]
    )


def test_setting_one_example_keeps_the_others_and_its_title() -> None:
    draft = number_range_draft()
    assert draft.set_example(example_rows("15"), index=None) == 0
    document = load_yaml(draft.to_yaml())
    document["examples"][0] |= {"title": "Ordinary", "description": "Inside range."}
    draft = MethodDraft(document, NUMBER_QUESTIONS, name="range", version=1)
    assert draft.set_example(example_rows("10"), index=None) == 1
    second = draft.examples[1]
    assert draft.set_example(example_rows("16"), index=0) == 0
    assert draft.examples[1] == second
    assert (draft.examples[0].title, draft.examples[0].description) == (
        "Ordinary",
        "Inside range.",
    )
    assert draft.examples[0].rows == example_rows("16")
    assert draft.set_example([], index=0) is None
    assert draft.examples == (second,)


def test_a_draft_reads_its_examples_without_results_and_keeps_those_it_is_given() -> None:
    draft = number_range_draft()
    correct_row = ExampleRow.model_validate(
        {"answers": {"Q1": "15"}, "expected": {"Q1": {"marks": 1, "feedback": "Correct."}}}
    )
    draft.set_example([correct_row], index=None)
    assert draft.examples[0].rows[0].expected == correct_row.expected
    reopened = MethodDraft.from_yaml(draft.to_yaml(), NUMBER_QUESTIONS, name="range", version=1)
    assert reopened.examples[0].rows == example_rows("15")


def test_published_example_settings_use_the_documented_yaml_alias() -> None:
    draft = number_range_draft()
    draft.bind_new_setting("/with/steps/0/with/min_value")
    draft.test_values = {"min_value": 12}
    draft.set_example(example_rows("15"), index=None)
    published = load_yaml(draft.to_yaml())
    assert published["examples"][0]["with"] == {"min_value": 12}
    assert "with_" not in published["examples"][0]


def test_examples_are_written_without_defaults_and_read_back() -> None:
    draft = number_range_draft()
    draft.set_example([ExampleRow(answers={"Q1": "15"})], index=None)
    [example] = load_yaml(draft.to_yaml())["examples"]
    assert example["questions"] == {"Q1": {"type": "number", "max_marks": 2.0}}
    assert "with" not in example
    assert MethodDraft.from_yaml(draft.to_yaml(), draft.questions, name="range", version=1).examples


def test_an_unfinished_draft_says_what_it_is_so_far() -> None:
    text = "label: Ranges\nsummary: Within ranges.\nsteps: [{uses: missing@1}]\n"
    draft = MethodDraft.from_yaml(
        text, NUMBER_QUESTIONS, name="sample/ranges", version=1, methods=Methods()
    )
    assert (draft.label, draft.summary) == ("Ranges", "Within ranges.")
    assert (draft.question_types, draft.marks_question_group) == (frozenset({"number"}), False)
    group = MethodDraft.from_yaml(
        "label: 3\nsteps: []\n",
        TWO_NUMBER_QUESTIONS,
        name="sample/pair",
        version=1,
        methods=Methods(),
    )
    assert (group.label, group.summary, group.question_types, group.marks_question_group) == (
        "",
        "",
        frozenset(),
        True,
    )


def test_the_details_form_chooses_question_types_one_by_one() -> None:
    question_types = number_range_draft().metadata_schema["properties"]["question_types"]
    assert question_types["type"] == "array"
    assert question_types["uniqueItems"] is True
    assert question_types["items"]["enum"] == ["text", "number", "choice", "values", "code"]
    assert question_types[HINT_KEY]["labels"]["values"] == "Multiple values"


def test_the_details_titles_tell_each_field_apart() -> None:
    titles = [
        property_schema["title"]
        for property_schema in number_range_draft().metadata_schema["properties"].values()
    ]
    assert len(titles) == len(set(titles))


def test_every_question_type_is_all() -> None:
    draft = number_range_draft()
    draft.update_metadata(question_types=["number", "text", "choice", "values", "code"])
    assert load_yaml(draft.to_yaml())["question_types"] == "all"
    assert draft.metadata["question_types"] == ["text", "number", "choice", "values", "code"]


def test_metadata_saved_as_it_is_leaves_the_file_unchanged() -> None:
    draft = number_range_draft()
    draft.update_metadata(
        question_types=["text", "number", "choice", "values", "code"], level="advanced"
    )
    before = draft.to_yaml()
    draft.update_metadata(**draft.metadata)
    assert draft.to_yaml() == before


def test_defaults_and_empty_optional_metadata_are_left_unset() -> None:
    draft = number_range_draft()
    before = load_yaml(draft.to_yaml())
    draft.update_metadata(guide="", level="basic", description="", feedback="")
    assert load_yaml(draft.to_yaml()) == before


def test_without_question_types_a_draft_marks_its_questions_type() -> None:
    draft = number_range_draft()
    draft.update_metadata(question_types=["number", "text"])
    draft.update_metadata(question_types=[])
    assert draft.question_types == {"number"}


def test_metadata_set_to_none_is_unset() -> None:
    draft = number_range_draft()
    draft.update_metadata(feedback="Well done.")
    draft.update_metadata(feedback=None)
    assert "feedback" not in draft.metadata
    assert "feedback" not in load_yaml(draft.to_yaml())


def test_a_name_a_setting_fills_labels_its_check_with_its_default() -> None:
    draft = number_range_draft()
    draft.steps_draft.update("/with/steps/0", {"name": "In range"})
    draft.bind_new_setting("/with/steps/0/name")
    assert draft.steps_draft.label("/with/steps/0") == "In range"
    draft.update_declared_setting("name_setting", {"title": "Check name"})
    assert draft.steps_draft.label("/with/steps/0") == "Number Range"


def test_a_field_a_setting_fills_has_its_problems_with_the_method() -> None:
    draft = MethodDraft.new(TEXT_QUESTIONS, name="answers", version=1, label="Answers")
    draft.steps_draft.add("/with/steps/0", "gradeflow/accepted-answers@1")
    draft.bind_new_setting("/with/steps/0/with/answers")
    assert draft.steps_draft.problems == []
    assert draft.problems
    with pytest.raises(EditingError):
        draft.steps_draft.finalized()


def test_a_change_to_a_check_thats_not_there_is_refused() -> None:
    draft = number_range_draft()
    with pytest.raises(EditingError):
        draft.steps_draft.update("/with/steps/5", {"min_value": 1})
