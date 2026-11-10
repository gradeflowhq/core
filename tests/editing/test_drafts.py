from collections.abc import Callable

import pytest

from gradeflow_core import Check, InvalidError, Problem, QuestionSet
from gradeflow_core.editing import HINT_KEY, Condition, ConditionRow, EditingError, MarkingDraft
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

from ..methods.helpers import located

QUESTIONS = QuestionSet(
    {"Q1": TextQuestion(), "Q2": ValuesQuestion(value_types=["number", "number"])}
)


def workflow_draft() -> MarkingDraft:
    draft = MarkingDraft.new("gradeflow/accepted-answers@1", QUESTIONS, "Q1")
    draft.change_method("", "gradeflow/workflow@1")
    draft.add("with.steps.1", "gradeflow/key-points@1")
    return draft


def test_a_new_marking_names_its_question() -> None:
    draft = MarkingDraft.new("gradeflow/accepted-answers@1", QUESTIONS, "Q1")
    assert draft.data == {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {}}
    with pytest.raises(EditingError):
        MarkingDraft.new("gradeflow/multiple-choice@1", QUESTIONS, "Q1")


def test_another_method_keeps_where_the_check_is_and_holds_the_old_one() -> None:
    draft = MarkingDraft.new("gradeflow/accepted-answers@1", QUESTIONS, "Q1")
    draft.update("", {"answers": ["Paris"]})
    draft.change_method("", "gradeflow/not-allowed@1")
    assert draft.data == {
        "question": "Q1",
        "uses": "gradeflow/not-allowed@1",
        "with": {"check": {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["Paris"]}}},
    }
    assert draft.finalized().dump() == draft.data


def test_another_method_keeps_the_steps_own_keys_in_one_order() -> None:
    draft = workflow_draft()
    data = draft.data
    uses = data["with"]["steps"][0]["uses"]
    data["with"]["steps"][0] = {"with": {}, "weight": 2, "uses": uses, "name": "Capital"}
    draft.replace(data)
    draft.change_method("with.steps.0", "gradeflow/pattern@1")
    assert list(draft.data["with"]["steps"][0]) == ["name", "weight", "uses", "with"]


def test_checks_are_added_moved_and_removed() -> None:
    draft = workflow_draft()
    assert [setting.step_paths for setting in draft.settings_holding_steps("")] == [
        ("with.steps.0", "with.steps.1")
    ]
    draft.move("with.steps.1", -1)
    assert [check["uses"] for check in draft.data["with"]["steps"]] == [
        "gradeflow/key-points@1",
        "gradeflow/accepted-answers@1",
    ]
    assert draft.remove("with.steps.0") == ""
    assert [check["uses"] for check in draft.data["with"]["steps"]] == [
        "gradeflow/accepted-answers@1"
    ]


def test_a_check_moves_past_the_others_and_stops_at_either_end() -> None:
    draft = workflow_draft()
    draft.add("with.steps.2", "gradeflow/full-marks@1")
    draft.move("with.steps.0", 2)
    assert [check["uses"] for check in draft.data["with"]["steps"]] == [
        "gradeflow/key-points@1",
        "gradeflow/full-marks@1",
        "gradeflow/accepted-answers@1",
    ]
    draft.move("with.steps.2", -5)
    assert draft.data["with"]["steps"][0]["uses"] == "gradeflow/accepted-answers@1"


def test_a_key_no_check_has_is_kept_and_is_a_problem() -> None:
    draft = workflow_draft()
    data = draft.data
    data["with"]["steps"][0]["nmae"] = "Typo"
    draft = MarkingDraft(data, QUESTIONS, "Q1")
    draft.update("with.steps.0", {"weight": 2})
    assert list(draft.data["with"]["steps"][0]) == ["nmae", "weight", "uses", "with"]
    assert ("with.steps.0", ("nmae",), "Unknown setting.") in located(draft.problems)


@pytest.mark.parametrize(
    "change",
    [
        lambda draft: draft.update("with.steps.5", {"points": ["x"]}),
        lambda draft: draft.change_method("with.steps.5", "gradeflow/not-allowed@1"),
        lambda draft: draft.move("with.steps.5", 1),
        lambda draft: draft.move("", 1),
        lambda draft: draft.remove("with.steps.5"),
        lambda draft: draft.remove(""),
        lambda draft: draft.add("with.steps.5", "gradeflow/key-points@1"),
        lambda draft: draft.add("with.nowhere.0", "gradeflow/key-points@1"),
    ],
    ids=[
        "update a missing check",
        "change a missing check's method",
        "move a missing check",
        "move the top check",
        "remove a missing check",
        "remove the top check",
        "add past the end",
        "add to a missing slot",
    ],
)
def test_a_change_to_a_check_or_slot_thats_not_there_is_refused(
    change: Callable[[MarkingDraft], object],
) -> None:
    draft = workflow_draft()
    before = draft.data
    with pytest.raises(EditingError):
        change(draft)
    assert draft.data == before


def test_a_check_for_each_value_is_fixed_and_reads_the_value_in_its_place() -> None:
    draft = MarkingDraft.new("gradeflow/multiple-values@1", QUESTIONS, "Q2")
    (setting,) = draft.settings_holding_steps("")
    assert (setting.minimum, setting.maximum, setting.is_full) == (2, 2, True)
    assert setting.step_paths == ("with.checks.0", "with.checks.1")
    with pytest.raises(EditingError):
        draft.add(setting.new_step_path, "gradeflow/accepted-answers@1")
    assert (draft.position("with.checks.1").value, draft.label("with.checks.1")) == (2, "Value 2")
    assert "input" not in draft.schema("with.checks.1")["properties"]


def test_fields_are_settings_and_keys_together_and_defaults_are_left_out() -> None:
    draft = workflow_draft()
    draft.update("with.steps.1", {"points": ["capital"]})
    draft.update("with.steps.0", {"answers": ["Paris"], "weight": 1, "name": "", "match": "exact"})
    assert draft.finalized().dump()["with"]["steps"][0] == {
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": ["Paris"]},
    }
    draft.update("with.steps.0", {"weight": 0.5})
    assert draft.values("with.steps.0")["weight"] == 0.5
    assert draft.values("with.steps.0")["answers"] == ["Paris"]
    assert list(draft.schema("with.steps.0")["properties"])[:2] == ["answers", "match"]


def test_optional_settings_left_empty_are_not_written() -> None:
    number_questions = QuestionSet({"Q1": NumberQuestion()})
    draft = MarkingDraft.new("gradeflow/number-range@1", number_questions, "Q1")
    draft.update("", {"min_value": 1, "max_value": None})
    assert draft.data["with"] == {"min_value": 1}
    pattern_draft = MarkingDraft.new("gradeflow/pattern@1", QUESTIONS, "Q1")
    pattern_draft.update("", {"pattern": ""})
    assert pattern_draft.data["with"] == {"pattern": ""}, (
        "a required setting stays as it was cleared"
    )
    code_questions = QuestionSet({"Q1": CodeQuestion()})
    code_uses_draft = MarkingDraft.new("gradeflow/code-uses@1", code_questions, "Q1")
    code_uses_draft.update("", {"looks_for": "loop", "function": "", "module": ""})
    assert code_uses_draft.data["with"] == {"looks_for": "loop"}
    assert code_uses_draft.problems == []
    defaulted_draft = workflow_draft()
    defaulted_draft.update("", {"marks_by": ""})
    defaulted_draft.update("with.steps.0", {"answers": ["x"], "match": None, "tolerance": None})
    assert defaulted_draft.data["with"]["steps"][0]["with"] == {"answers": ["x"]}
    assert "marks_by" not in defaulted_draft.data["with"]
    custom_code_draft = MarkingDraft.new("gradeflow/custom-code@1", code_questions, "Q1")
    custom_code_draft.update("", {"code": ""})
    assert custom_code_draft.data["with"] == {"code": ""}


def test_a_condition_names_the_checks_it_reads_by_path_and_gives_them_ids() -> None:
    draft = workflow_draft()
    draft.update("with.steps.1", {"if": Condition((ConditionRow("with.steps.0", "incorrect"),))})
    first, second = draft.data["with"]["steps"]
    assert first["id"] == "accepted_answers"
    assert second["if"] == "{{ not steps.accepted_answers.output.correct }}"
    assert draft.values("with.steps.1")["if"] == Condition(
        (ConditionRow("with.steps.0", "incorrect"),)
    )
    assert draft.earlier_checks("with.steps.1") == ["with.steps.0"]


def test_a_condition_is_offered_where_it_has_an_earlier_check_to_read() -> None:
    draft = workflow_draft()
    draft.update("with.steps.0", {"answers": ["Paris"]})
    draft.update("with.steps.1", {"points": ["capital"]})
    assert "if" not in draft.schema("")["properties"]
    assert "if" not in draft.schema("with.steps.0")["properties"]
    assert "if" in draft.schema("with.steps.1")["properties"]
    draft.update("with.steps.1", {"if": Condition((ConditionRow("with.steps.0", "correct"),))})
    draft.move("with.steps.1", -1)
    assert "if" in draft.schema("with.steps.0")["properties"]
    assert located(draft.problems) == [
        ("with.steps.0", ("if",), "Accepted Answers must come before this step.")
    ]


def test_checks_inside_a_check_that_chose_the_question_or_value_dont_choose_again() -> None:
    accepted_answers_check = {
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": ["x"]},
    }
    workflow = {"uses": "gradeflow/workflow@1", "with": {"steps": [accepted_answers_check]}}
    group_marking = {
        "uses": "gradeflow/workflow@1",
        "with": {
            "steps": [
                {**accepted_answers_check, "question": "Q1"},
                {**workflow, "question": "Q1"},
            ]
        },
    }
    draft = MarkingDraft(group_marking, QUESTIONS, None)
    assert "question" in draft.schema("with.steps.0")["properties"]
    assert "question" not in draft.schema("with.steps.1.with.steps.0")["properties"]
    number_range = {"uses": "gradeflow/number-range@1", "with": {"min_value": 1}}
    values_workflow = {
        "uses": "gradeflow/workflow@1",
        "with": {"steps": [number_range, number_range]},
    }
    draft = MarkingDraft(
        {
            "question": "Q2",
            "uses": "gradeflow/workflow@1",
            "with": {"steps": [number_range, {**values_workflow, "input": "{{ values[1] }}"}]},
        },
        QUESTIONS,
        "Q2",
    )
    assert "input" in draft.schema("with.steps.0")["properties"]
    assert "input" not in draft.schema("with.steps.1.with.steps.0")["properties"]


def test_a_condition_reading_a_later_check_is_reported_while_another_check_is_unfinished() -> None:
    draft = workflow_draft()
    draft.update("with.steps.0", {"answers": ["Paris"]})
    draft.update("with.steps.1", {"if": Condition((ConditionRow("with.steps.0", "correct"),))})
    draft.move("with.steps.1", -1)
    assert ("with.steps.0", ("if",), "Accepted Answers must come before this step.") in located(
        draft.problems
    )


def test_a_condition_reading_a_removed_check_says_no_check_before_it_has_that_name() -> None:
    draft = workflow_draft()
    draft.update("with.steps.0", {"answers": ["Paris"], "name": "Capital"})
    draft.update("with.steps.1", {"points": ["capital"]})
    draft.update("with.steps.1", {"if": Condition((ConditionRow("with.steps.0", "correct"),))})
    draft.remove("with.steps.0")
    assert located(draft.problems) == [
        ("with.steps.0", ("if",), "No step before this one is called capital.")
    ]


def test_checks_in_nested_lists_read_those_before_them_at_every_level() -> None:
    full_marks = {"uses": "gradeflow/full-marks@1"}
    marking = {
        "question": "Q1",
        "uses": "gradeflow/workflow@1",
        "with": {
            "steps": [
                full_marks,
                {
                    "uses": "gradeflow/workflow@1",
                    "with": {
                        "steps": [
                            full_marks,
                            {**full_marks, "if": "{{ steps.last.output.correct }}"},
                        ]
                    },
                },
                {**full_marks, "id": "last"},
            ]
        },
    }
    draft = MarkingDraft(marking, QUESTIONS, "Q1")
    assert draft.earlier_checks("with.steps.1.with.steps.1") == [
        "with.steps.0",
        "with.steps.1.with.steps.0",
    ]
    assert [(problem.step_path, problem.message) for problem in draft.problems] == [
        ("with.steps.1.with.steps.1", "Full Marks must come before this step.")
    ]


def test_a_condition_other_than_rows_stays_text() -> None:
    draft = workflow_draft()
    draft.update("with.steps.1", {"if": "{{ steps.accepted_answers.output.proportion > 0.2 }}"})
    assert (
        draft.values("with.steps.1")["if"] == "{{ steps.accepted_answers.output.proportion > 0.2 }}"
    )


def test_problems_are_on_the_check_and_field_at_fault() -> None:
    draft = workflow_draft()
    assert [
        (problem.step_path, problem.field, problem.is_incomplete) for problem in draft.problems
    ] == [
        ("with.steps.0", ("answers",), True),
        ("with.steps.1", ("points",), True),
    ]
    with pytest.raises(EditingError, match=r"^answers: "):
        draft.finalized()
    draft.update("with.steps.0", {"answers": ["Paris"]})
    draft.update("with.steps.1", {"points": ["capital"]})
    assert draft.problems == []
    assert draft.finalized().questions_marked() == {"Q1"}
    assert draft.describe("with.steps.0") == "Matches `Paris`."


def test_problems_are_where_validating_the_check_puts_them() -> None:
    draft = MarkingDraft.new("gradeflow/custom-code@1", QUESTIONS, "Q1")
    draft.update(
        "",
        {"code": "proportion = 1", "parameters": {"n": {"type": "integer", "value": "x"}}},
    )
    with pytest.raises(InvalidError) as exc_info:
        Check.load(draft.data)
    assert (
        draft.problems
        == exc_info.value.problems
        == [Problem("Must be a whole number.", field=("parameters", "n", "value"))]
    )


def test_a_check_is_labelled_by_its_name_or_else_its_method() -> None:
    draft = workflow_draft()
    draft.update("with.steps.0", {"name": "Capital"})
    assert draft.label("with.steps.0") == "Capital"
    method = draft.method("with.steps.1")
    assert method is not None
    assert draft.label("with.steps.1") == method.label
    with pytest.raises(EditingError):
        draft.position("marks_by")


def test_notes_say_what_a_checks_place_does_to_it() -> None:
    draft = workflow_draft()
    assert draft.notes("with.steps.1") == []
    draft.update("with.steps.0", {"name": "Capital", "weight": 0.5})
    incorrect = Condition((ConditionRow("with.steps.0", "incorrect"),))
    draft.update("with.steps.1", {"weight": -1, "if": incorrect})
    assert draft.notes("with.steps.0") == ["Worth ½"]
    assert draft.notes("with.steps.1") == ["Penalty 1", "Only when Capital isn't correct"]
    draft.update("with.steps.1", {"if": "{{ steps.capital.output.proportion > 0.2 }}"})
    assert draft.notes("with.steps.1")[-1] == "Only when its condition holds"


def test_a_weight_zero_check_notes_that_it_gives_no_marks() -> None:
    draft = MarkingDraft(
        {
            "uses": "gradeflow/workflow@1",
            "with": {
                "steps": [
                    {
                        "question": "Q1",
                        "uses": "gradeflow/accepted-answers@1",
                        "weight": 0,
                        "with": {"answers": ["Paris"]},
                    }
                ]
            },
        },
        QUESTIONS,
        None,
    )
    assert draft.notes("with.steps.0") == ["Q1", "Gives no marks"]


def test_a_weight_that_isnt_a_number_has_no_note() -> None:
    data = workflow_draft().data
    data["with"]["steps"][1] |= {"weight": "heavy", "if": "{{ steps.x.output.correct }}"}
    draft = MarkingDraft(data, QUESTIONS, "Q1")
    assert draft.notes("with.steps.1") == ["Only when its condition holds"]


def test_a_step_reads_the_output_of_the_step_of_that_id_in_its_own_workflow() -> None:
    def workflow(name: str, extract_settings: dict[str, str]) -> dict[str, object]:
        extract = {"id": "part", "name": name, "uses": "gradeflow/extract@1"}
        accepted_answers = {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["1"]}}
        steps = [{**extract, "with": extract_settings}, accepted_answers]
        return {"uses": "gradeflow/workflow@1", "with": {"steps": steps}}

    number, unit = workflow("Number", {}), workflow("Unit", {"type": "text", "pattern": r"(\w+)$"})
    marking = {"question": "Q1", "uses": "gradeflow/workflow@1", "with": {"steps": [number, unit]}}
    draft = MarkingDraft(marking, QUESTIONS, "Q1")
    for index, name in enumerate(["Number", "Unit"]):
        input_labels = draft.schema(f"with.steps.{index}.with.steps.1")["properties"]["input"][
            HINT_KEY
        ]["labels"]
        assert input_labels["{{ steps.part.output }}"] == f"Output of {name}"


@pytest.mark.parametrize("settings", ["x", 3, ["a"]])
def test_settings_that_arent_a_mapping_are_a_problem(settings: object) -> None:
    steps = [
        {"id": "part", "uses": "gradeflow/extract@1", "with": settings},
        {"uses": "gradeflow/workflow@1", "with": settings},
        {"uses": "gradeflow/accepted-answers@1", "with": settings},
    ]
    marking = {"question": "Q1", "uses": "gradeflow/workflow@1", "with": {"steps": steps}}
    draft = MarkingDraft(marking, QUESTIONS, "Q1")
    assert draft.paths() == ["", "with.steps.0", "with.steps.1", "with.steps.2"]
    for path in draft.paths():
        assert draft.schema(path)
        assert draft.methods(path)
        draft.values(path)
        draft.notes(path)
        draft.describe(path)
    assert draft.answer_decisions() == {}
    assert draft.problems


def test_a_condition_that_cant_be_read_says_so() -> None:
    draft = workflow_draft()
    draft.update("with.steps.0", {"answers": ["Paris"]})
    draft.update(
        "with.steps.1", {"points": ["capital"], "if": "{{ steps.x.output.correct garbage(( }}"}
    )
    assert located(draft.problems) == [("with.steps.1", ("if",), "Can't read this condition.")]


def test_a_method_that_cant_mark_the_question_says_so_before_its_finished() -> None:
    draft = MarkingDraft(
        {"question": "Q2", "uses": "gradeflow/accepted-answers@1", "with": {}}, QUESTIONS, "Q2"
    )
    assert [
        (problem.step_path, problem.field, problem.is_incomplete) for problem in draft.problems
    ] == [("", (), False)]
    assert draft.problems[0].message.startswith("Accepted Answers can't be used on")


def test_a_checks_values_are_read_as_its_question_reads_them() -> None:
    choice_questions = QuestionSet({"Q1": ChoiceQuestion(options=("a", "b"))})
    marking = {"question": "Q1", "uses": "gradeflow/multiple-choice@1", "with": {"correct": ["B"]}}
    draft = MarkingDraft(marking, choice_questions, "Q1")
    assert draft.values("")["correct"] == ["b"]
    assert draft.data["with"]["correct"] == ["B"], "as written until it is changed"


def test_a_value_named_uses_isnt_a_check() -> None:
    draft = workflow_draft()
    draft.add("with.steps.2", "gradeflow/custom-code@1")
    draft.update("with.steps.2", {"parameters": {"uses": {"type": "text", "value": "x"}}})
    assert draft.paths() == ["", "with.steps.0", "with.steps.1", "with.steps.2"]
    assert not draft.is_step("with.steps.2.with.parameters")
    with pytest.raises(EditingError):
        draft.remove("with.steps.2.with.parameters")
    assert draft.values("with.steps.2")["parameters"] == {"uses": {"type": "text", "value": "x"}}


def draft_from_yaml(text: str, question_id: str | None = "Q1") -> MarkingDraft:
    return MarkingDraft.from_yaml(text, QUESTIONS, question_id)


def refused_yaml_problems(text: str, question_id: str | None = "Q1") -> list[str]:
    with pytest.raises(InvalidError) as exc_info:
        draft_from_yaml(text, question_id)
    return [str(problem) for problem in exc_info.value.problems]


@pytest.mark.parametrize(
    "text",
    [
        "question: Q2\nuses: gradeflow/workflow@1\n",
        "- uses: gradeflow/workflow@1\n",
        "uses: [\n",
    ],
)
def test_yaml_that_isnt_a_marking_of_the_question_is_refused(text: str) -> None:
    assert refused_yaml_problems(text)


@pytest.mark.parametrize(
    "text",
    [
        "question: Q1\nuses: gradeflow/workflow@1\nwith: {steps: []}\n",
        "question: Q1\nuses: gradeflow/nope@1\n",
    ],
)
def test_a_marking_of_the_question_is_read_problems_and_all(text: str) -> None:
    assert draft_from_yaml(text).problems


def test_a_question_groups_marking_names_no_question() -> None:
    assert refused_yaml_problems(
        "question: Q1\nuses: gradeflow/workflow@1\nwith: {steps: []}\n", None
    ) == [
        "Line 1: Remove question; a question group's checks name their own.",
        "Line 3: steps: Add at least one step.",
    ]


def test_a_marking_needs_not_name_its_question() -> None:
    draft = draft_from_yaml("uses: gradeflow/accepted-answers@1\nwith: {answers: [a]}\n")
    assert draft.problems == []
    assert draft.finalized().question == "Q1"


def test_every_problem_is_on_its_line_even_with_the_wrong_question() -> None:
    messages = refused_yaml_problems(
        "question: Q2\nuses: gradeflow/accepted-answers@1\nwith: {answers: [a], match: fuzzy}\n"
    )
    assert messages[0] == "Line 1: Set question to Q1."
    assert messages[1].startswith("Line 3: ")


def test_a_changed_marking_says_its_problems_as_any_draft_does() -> None:
    draft = draft_from_yaml("question: Q1\nuses: gradeflow/workflow@1\nwith: {steps: []}\n")
    assert [str(problem) for problem in draft.problems] == ["Line 3: steps: Add at least one step."]
    draft.add("with.steps.0", "gradeflow/key-points@1")
    assert draft.problems == MarkingDraft(draft.data, QUESTIONS, "Q1").problems
    assert draft.problems


def test_editing_a_new_step_leaves_its_methods_starting_settings_alone() -> None:
    def answer_sets_draft() -> MarkingDraft:
        draft = MarkingDraft.new("gradeflow/workflow@1", QUESTIONS, None)
        draft.add("with.steps.0", "gradeflow/answer-sets@1")
        return draft

    draft = answer_sets_draft()
    starting_settings = draft.data["with"]["steps"][0]["with"]
    draft.add("with.steps.0.with.sets.0.with.steps.0", "gradeflow/accepted-answers@1")
    assert answer_sets_draft().data["with"]["steps"][0]["with"] == starting_settings


@pytest.mark.parametrize("steps", ["5", "x"])
def test_steps_that_arent_a_list_hold_no_steps(steps: str) -> None:
    draft = draft_from_yaml(f"uses: gradeflow/workflow@1\nwith: {{steps: {steps}}}\n")
    (setting,) = draft.settings_holding_steps("")
    assert setting.step_paths == ()
    draft.add(setting.new_step_path, "gradeflow/accepted-answers@1")
    assert [step["uses"] for step in draft.values("")["steps"]] == ["gradeflow/accepted-answers@1"]
