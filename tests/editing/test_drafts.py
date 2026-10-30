from collections.abc import Callable
from typing import Any

import pytest

from gradeflow_core import Check, InvalidError, Method, Problem, QuestionSet
from gradeflow_core.editing import Condition, ConditionRow, EditingError, MarkingDraft
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.code import CodeQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion

QUESTIONS = QuestionSet(
    {"Q1": TextQuestion(), "Q2": ValuesQuestion(value_types=["number", "number"])}
)


def combination() -> MarkingDraft:
    draft = MarkingDraft.new("gradeflow/accepted-answers@1", QUESTIONS, "Q1")
    draft.retype("", "gradeflow/combination@1")
    draft.add("with.checks.1", "gradeflow/key-points@1")
    return draft


def test_a_new_marking_names_its_question() -> None:
    draft = MarkingDraft.new("gradeflow/accepted-answers@1", QUESTIONS, "Q1")
    assert draft.data == {"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {}}
    with pytest.raises(EditingError):
        MarkingDraft.new("gradeflow/multiple-choice@1", QUESTIONS, "Q1")


def test_another_method_keeps_where_the_check_is_and_holds_the_old_one() -> None:
    draft = MarkingDraft.new("gradeflow/accepted-answers@1", QUESTIONS, "Q1")
    draft.update("", {"answers": ["Paris"]})
    draft.retype("", "gradeflow/not-allowed@1")
    assert draft.data == {
        "question": "Q1",
        "uses": "gradeflow/not-allowed@1",
        "with": {"check": {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["Paris"]}}},
    }
    assert draft.finalized().dump() == draft.data


def test_checks_are_added_moved_and_removed() -> None:
    draft = combination()
    assert [slot.checks for slot in draft.slots("")] == [("with.checks.0", "with.checks.1")]
    draft.move("with.checks.1", -1)
    assert [check["uses"] for check in draft.data["with"]["checks"]] == [
        "gradeflow/key-points@1",
        "gradeflow/accepted-answers@1",
    ]
    assert draft.remove("with.checks.0") == ""
    assert [check["uses"] for check in draft.data["with"]["checks"]] == [
        "gradeflow/accepted-answers@1"
    ]


@pytest.mark.parametrize(
    "change",
    [
        lambda draft: draft.update("with.checks.5", {"points": ["x"]}),
        lambda draft: draft.retype("with.checks.5", "gradeflow/not-allowed@1"),
        lambda draft: draft.move("with.checks.5", 1),
        lambda draft: draft.move("", 1),
        lambda draft: draft.remove("with.checks.5"),
        lambda draft: draft.remove(""),
        lambda draft: draft.add("with.checks.5", "gradeflow/key-points@1"),
        lambda draft: draft.add("with.nowhere.0", "gradeflow/key-points@1"),
    ],
)
def test_a_change_to_a_check_or_slot_thats_not_there_is_refused(
    change: Callable[[MarkingDraft], object],
) -> None:
    draft = combination()
    before = draft.data
    with pytest.raises(EditingError):
        change(draft)
    assert draft.data == before


def test_a_check_for_each_value_is_fixed_and_reads_the_value_in_its_place() -> None:
    draft = MarkingDraft.new("gradeflow/multiple-values@1", QUESTIONS, "Q2")
    (slot,) = draft.slots("")
    assert (slot.minimum, slot.maximum, slot.full) == (2, 2, True)
    assert slot.checks == ("with.checks.0", "with.checks.1")
    with pytest.raises(EditingError):
        draft.add(slot.new, "gradeflow/accepted-answers@1")
    assert (draft.position("with.checks.1").value, draft.label("with.checks.1")) == (2, "Value 2")
    assert "value" not in draft.schema("with.checks.1")["properties"]


def test_fields_are_settings_and_keys_together_and_defaults_are_left_out() -> None:
    draft = combination()
    draft.update("with.checks.1", {"points": ["capital"]})
    draft.update("with.checks.0", {"answers": ["Paris"], "weight": 1, "name": "", "match": "exact"})
    assert draft.finalized().dump()["with"]["checks"][0] == {
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": ["Paris"]},
    }
    draft.update("with.checks.0", {"weight": 0.5})
    assert draft.values("with.checks.0")["weight"] == 0.5
    assert draft.values("with.checks.0")["answers"] == ["Paris"]
    assert list(draft.schema("with.checks.0")["properties"])[:2] == ["answers", "match"]


def test_optional_settings_left_empty_are_not_written() -> None:
    number = QuestionSet({"Q1": NumberQuestion()})
    draft = MarkingDraft.new("gradeflow/number-range@1", number, "Q1")
    draft.update("", {"min_value": 1, "max_value": None})
    assert draft.data["with"] == {"min_value": 1}
    pattern = MarkingDraft.new("gradeflow/pattern@1", QUESTIONS, "Q1")
    pattern.update("", {"pattern": ""})
    assert pattern.data["with"] == {"pattern": ""}, "a required setting stays as it was cleared"
    code = QuestionSet({"Q1": CodeQuestion()})
    uses = MarkingDraft.new("gradeflow/code-uses@1", code, "Q1")
    uses.update("", {"looks_for": "loop", "function": "", "module": ""})
    assert uses.data["with"] == {"looks_for": "loop"}
    assert uses.problems == []


def test_a_condition_names_the_checks_it_reads_by_path_and_gives_them_ids() -> None:
    draft = combination()
    draft.update("with.checks.1", {"if": Condition((ConditionRow("with.checks.0", "incorrect"),))})
    first, second = draft.data["with"]["checks"]
    assert first["id"] == "accepted_answers"
    assert second["if"] == "{{ not checks.accepted_answers.correct }}"
    assert draft.values("with.checks.1")["if"] == Condition(
        (ConditionRow("with.checks.0", "incorrect"),)
    )
    assert draft.earlier("with.checks.1") == ["with.checks.0"]


def test_a_condition_is_offered_where_it_has_an_earlier_check_to_read() -> None:
    draft = combination()
    draft.update("with.checks.0", {"answers": ["Paris"]})
    draft.update("with.checks.1", {"points": ["capital"]})
    assert "if" not in draft.schema("")["properties"]
    assert "if" not in draft.schema("with.checks.0")["properties"]
    assert "if" in draft.schema("with.checks.1")["properties"]
    draft.update("with.checks.1", {"if": Condition((ConditionRow("with.checks.0", "correct"),))})
    draft.move("with.checks.1", -1)
    assert "if" in draft.schema("with.checks.0")["properties"]
    assert [(p.check, p.field, p.message) for p in draft.problems] == [
        ("with.checks.0", ("if",), "Accepted Answers must come before this check.")
    ]


def test_checks_inside_a_check_that_chose_the_question_or_value_dont_choose_again() -> None:
    answers = {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["x"]}}
    combined = {"uses": "gradeflow/combination@1", "with": {"checks": [answers]}}
    group = {
        "uses": "gradeflow/combination@1",
        "with": {"checks": [{**answers, "question": "Q1"}, {**combined, "question": "Q1"}]},
    }
    draft = MarkingDraft(group, QUESTIONS, None)
    assert "question" in draft.schema("with.checks.0")["properties"]
    assert "question" not in draft.schema("with.checks.1.with.checks.0")["properties"]
    numbers = {"uses": "gradeflow/number-range@1", "with": {"min_value": 1}}
    values = {"uses": "gradeflow/combination@1", "with": {"checks": [numbers, numbers]}}
    draft = MarkingDraft(
        {**values, "question": "Q2", "with": {"checks": [numbers, {**values, "value": 2}]}},
        QUESTIONS,
        "Q2",
    )
    assert "value" in draft.schema("with.checks.0")["properties"]
    assert "value" not in draft.schema("with.checks.1.with.checks.0")["properties"]


def test_a_condition_reading_a_later_check_is_said_while_another_check_is_unfinished() -> None:
    draft = combination()
    draft.update("with.checks.0", {"answers": ["Paris"]})
    draft.update("with.checks.1", {"if": Condition((ConditionRow("with.checks.0", "correct"),))})
    draft.move("with.checks.1", -1)
    assert ("with.checks.0", ("if",), "Accepted Answers must come before this check.") in [
        (p.check, p.field, p.message) for p in draft.problems
    ]


def test_a_condition_reading_a_removed_check_says_no_check_before_it_has_that_name() -> None:
    draft = combination()
    draft.update("with.checks.0", {"answers": ["Paris"], "name": "Capital"})
    draft.update("with.checks.1", {"points": ["capital"]})
    draft.update("with.checks.1", {"if": Condition((ConditionRow("with.checks.0", "correct"),))})
    draft.remove("with.checks.0")
    assert [(p.check, p.field, p.message) for p in draft.problems] == [
        ("with.checks.0", ("if",), "No check before this one is called capital.")
    ]


def test_checks_in_nested_lists_read_those_before_them_at_every_level() -> None:
    full = {"uses": "gradeflow/full-marks@1"}
    data = {
        "question": "Q1",
        "uses": "gradeflow/combination@1",
        "with": {
            "checks": [
                full,
                {
                    "uses": "gradeflow/combination@1",
                    "with": {"checks": [full, {**full, "if": "{{ checks.last.correct }}"}]},
                },
                {**full, "id": "last"},
            ]
        },
    }
    draft = MarkingDraft(data, QUESTIONS, "Q1")
    assert draft.earlier("with.checks.1.with.checks.1") == [
        "with.checks.0",
        "with.checks.1.with.checks.0",
    ]
    assert [(p.check, p.message) for p in draft.problems] == [
        ("with.checks.1.with.checks.1", "Full Marks must come before this check.")
    ]


def test_a_condition_other_than_rows_stays_text() -> None:
    draft = combination()
    draft.update("with.checks.1", {"if": "{{ checks.accepted_answers.proportion > 0.2 }}"})
    assert draft.values("with.checks.1")["if"] == "{{ checks.accepted_answers.proportion > 0.2 }}"


def test_problems_are_on_the_check_and_field_at_fault() -> None:
    draft = combination()
    assert [(p.check, p.field, p.incomplete) for p in draft.problems] == [
        ("with.checks.0", ("answers",), True),
        ("with.checks.1", ("points",), True),
    ]
    with pytest.raises(EditingError, match=r"^answers: "):
        draft.finalized()
    draft.update("with.checks.0", {"answers": ["Paris"]})
    draft.update("with.checks.1", {"points": ["capital"]})
    assert draft.problems == []
    assert draft.finalized().questions_marked() == {"Q1"}
    assert draft.describe("with.checks.0") == "Matches `Paris`."


def test_problems_are_where_validating_the_check_puts_them() -> None:
    draft = MarkingDraft.new("gradeflow/custom-code@1", QUESTIONS, "Q1")
    draft.update(
        "",
        {"code": "proportion = 1", "parameters": {"n": {"type": "integer", "value": "x"}}},
    )
    with pytest.raises(InvalidError) as caught:
        Check.load(draft.data)
    assert (
        draft.problems
        == caught.value.problems
        == [Problem("Must be a whole number.", field=("parameters", "n", "value"))]
    )


def test_a_check_is_labelled_by_its_name_or_else_its_method() -> None:
    draft = combination()
    draft.update("with.checks.0", {"name": "Capital"})
    assert draft.label("with.checks.0") == "Capital"
    method = draft.method("with.checks.1")
    assert method is not None
    assert draft.label("with.checks.1") == method.label
    with pytest.raises(EditingError):
        draft.position("with.combine")


def test_notes_say_what_a_checks_place_does_to_it() -> None:
    draft = combination()
    assert draft.notes("with.checks.1") == []
    draft.update("with.checks.0", {"name": "Capital", "weight": 0.5})
    incorrect = Condition((ConditionRow("with.checks.0", "incorrect"),))
    draft.update("with.checks.1", {"weight": -1, "if": incorrect})
    assert draft.notes("with.checks.0") == ["Worth ½"]
    assert draft.notes("with.checks.1") == ["Takes off 1", "Only when Capital isn't correct"]
    draft.update("with.checks.1", {"if": "{{ checks.capital.proportion > 0.2 }}"})
    assert draft.notes("with.checks.1")[-1] == "Only when its condition holds"


def test_a_read_only_check_notes_no_weight() -> None:
    draft = MarkingDraft(
        {
            "uses": "gradeflow/combination@1",
            "with": {
                "checks": [
                    {
                        "question": "Q1",
                        "uses": "gradeflow/accepted-answers@1",
                        "read_only": True,
                        "weight": -1,
                        "with": {"answers": ["Paris"]},
                    }
                ]
            },
        },
        QUESTIONS,
        None,
    )
    assert draft.notes("with.checks.0") == ["Q1", "Read only"]


def test_a_condition_that_cant_be_read_says_so() -> None:
    draft = combination()
    draft.update("with.checks.0", {"answers": ["Paris"]})
    draft.update("with.checks.1", {"points": ["capital"], "if": "{{ checks.x.correct garbage(( }}"})
    assert [(p.check, p.field, p.message) for p in draft.problems] == [
        ("with.checks.1", ("if",), "Can't read this condition.")
    ]


def test_a_method_that_cant_mark_the_question_says_so_before_its_finished() -> None:
    draft = MarkingDraft(
        {"question": "Q2", "uses": "gradeflow/accepted-answers@1", "with": {}}, QUESTIONS, "Q2"
    )
    assert [(p.check, p.field, p.incomplete) for p in draft.problems] == [("", (), False)]
    assert draft.problems[0].message.startswith("Accepted Answers can't mark")


def test_a_checks_schema_is_worked_out_once_for_where_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gradeflow_core.editing import drafts

    asked: list[str] = []
    real = drafts._check_schema

    def asking(method: Method[Any], *args: object, **kwargs: object) -> object:
        asked.append(method.name)
        return real(method, *args, **kwargs)  # type: ignore[arg-type]

    draft = combination()
    monkeypatch.setattr(drafts, "_check_schema", asking)
    for _ in range(2):
        draft.schema("with.checks.0")
        draft.schema("with.checks.1")
    assert asked == ["gradeflow/accepted-answers", "gradeflow/key-points"]


def test_a_checks_values_are_read_as_its_question_reads_them() -> None:
    choice = QuestionSet({"Q1": ChoiceQuestion(options=frozenset({"a", "b"}))})
    marking = {"question": "Q1", "uses": "gradeflow/multiple-choice@1", "with": {"correct": ["B"]}}
    draft = MarkingDraft(marking, choice, "Q1")
    assert draft.values("")["correct"] == ["b"]
    assert draft.data["with"]["correct"] == ["B"], "as written until it is changed"


def test_a_value_named_uses_isnt_a_check() -> None:
    draft = combination()
    draft.add("with.checks.2", "gradeflow/custom-code@1")
    draft.update("with.checks.2", {"parameters": {"uses": {"type": "text", "value": "x"}}})
    assert draft.paths() == ["", "with.checks.0", "with.checks.1", "with.checks.2"]
    assert not draft.is_check("with.checks.2.with.parameters")
    with pytest.raises(EditingError):
        draft.remove("with.checks.2.with.parameters")
    assert draft.values("with.checks.2")["parameters"] == {"uses": {"type": "text", "value": "x"}}


def written(text: str, question: str | None = "Q1") -> MarkingDraft:
    return MarkingDraft.from_yaml(text, QUESTIONS, question)


def refusal(text: str, question: str | None = "Q1") -> list[str]:
    with pytest.raises(InvalidError) as error:
        written(text, question)
    return [str(problem) for problem in error.value.problems]


@pytest.mark.parametrize(
    "text",
    [
        "question: Q2\nuses: gradeflow/combination@1\n",
        "- uses: gradeflow/combination@1\n",
        "uses: [\n",
    ],
)
def test_yaml_that_isnt_a_marking_of_the_question_is_refused(text: str) -> None:
    assert refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "question: Q1\nuses: gradeflow/combination@1\nwith: {checks: []}\n",
        "question: Q1\nuses: gradeflow/nope@1\n",
    ],
)
def test_a_marking_of_the_question_is_read_problems_and_all(text: str) -> None:
    assert written(text).problems


def test_a_question_groups_marking_names_no_question() -> None:
    assert refusal("question: Q1\nuses: gradeflow/combination@1\n", None) == [
        "Line 1: Remove question; a question group's checks name their own.",
        "checks: Add at least one check.",
    ]


def test_a_marking_needs_not_name_its_question() -> None:
    draft = written("uses: gradeflow/accepted-answers@1\nwith: {answers: [a]}\n")
    assert draft.problems == []
    assert draft.finalized().question == "Q1"


def test_every_problem_is_on_its_line_even_with_the_wrong_question() -> None:
    said = refusal(
        "question: Q2\nuses: gradeflow/accepted-answers@1\nwith: {answers: [a], match: fuzzy}\n"
    )
    assert said[0] == "Line 1: Set question to Q1."
    assert said[1].startswith("Line 3: ")


def test_a_changed_marking_says_its_problems_as_any_draft_does() -> None:
    draft = written("question: Q1\nuses: gradeflow/combination@1\nwith:\n  checks: []\n")
    assert [str(problem) for problem in draft.problems] == [
        "Line 4: checks: Add at least one check."
    ]
    draft.add("with.checks.0", "gradeflow/key-points@1")
    assert draft.problems == MarkingDraft(draft.data, QUESTIONS, "Q1").problems
    assert draft.problems
