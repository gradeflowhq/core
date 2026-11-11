from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from gradeflow_core import (
    BUILTINS,
    InvalidError,
    MarkScheme,
    MethodFileError,
    Methods,
    NumberQuestion,
    QuestionResult,
    QuestionSet,
    Submission,
    TextQuestion,
    ValuesQuestion,
    mark,
)
from gradeflow_core.editing import HINT_KEY, ExampleRow, MarkingDraft, MethodDraft, mark_rows
from gradeflow_core.methods.text.key_point import KeyPoint, KeyPointSettings
from gradeflow_core.questions.reading import TextReading
from gradeflow_core.questions.types import QuestionId
from gradeflow_core.steps.context import Context
from gradeflow_core.steps.outcome import QuestionOutcomes
from gradeflow_core.steps.running import transform
from gradeflow_core.steps.step import valid_step

from ..methods.helpers import workflow

EXTRACT = "gradeflow/extract@1"
KEY_POINTS = "gradeflow/key-points@1"
NUMBER_RANGE = "gradeflow/number-range@1"
ACCEPTED_ANSWERS = "gradeflow/accepted-answers@1"
PATTERN = "gradeflow/pattern@1"
QUESTIONS = QuestionSet(
    {
        "Q1": TextQuestion(max_marks=6),
        "Q2": ValuesQuestion(value_types=["number", "text"], max_marks=2),
        "Q3": NumberQuestion(),
    }
)
EXTRACT_LAST_NUMBER = {"id": "last_number", "uses": EXTRACT, "with": {"occurrence": "last"}}
LAST_NUMBER_IN_RANGE = {
    "input": "{{ steps.last_number.output }}",
    "uses": NUMBER_RANGE,
    "weight": 2,
    "with": {"min_value": 41.5, "max_value": 42.5},
}
REASONING_AND_NUMBER_MARKING: dict[str, Any] = {
    "question": "Q1",
    "uses": "gradeflow/workflow@1",
    "with": {
        "marks_by": "total",
        "steps": [
            EXTRACT_LAST_NUMBER,
            {"uses": KEY_POINTS, "weight": 4, "with": {"points": ["formula"]}},
            LAST_NUMBER_IN_RANGE,
        ],
    },
}


def result_of(
    marking: Mapping[str, Any], question_id: str = "Q1", **answers: str
) -> QuestionResult:
    scheme = MarkScheme.load([marking])
    assert scheme.problems(QUESTIONS) == []
    [marked_submission] = mark(QUESTIONS, scheme, [Submission(student_id="s1", answers=answers)])
    return marked_submission.results[question_id]


def problems_of(marking: Mapping[str, Any]) -> list[str]:
    return [problem.message for problem in MarkScheme.load([marking]).problems(QUESTIONS)]


def test_a_check_marks_what_a_transform_read() -> None:
    assert result_of(REASONING_AND_NUMBER_MARKING, Q1="By the formula, x = 42").marks == 6
    assert result_of(REASONING_AND_NUMBER_MARKING, Q1="By the formula, x = 7").marks == 4
    assert result_of(REASONING_AND_NUMBER_MARKING, Q1="I guess 42").marks == 2


def test_the_breakdown_shows_what_a_check_read_and_no_row_for_the_transform() -> None:
    breakdown = result_of(REASONING_AND_NUMBER_MARKING, Q1="By the formula, x = 42").breakdown
    assert breakdown is not None
    assert len(breakdown.children) == 2
    [read_as_detail] = breakdown.children[1].details
    assert (read_as_detail.title, read_as_detail.value) == ("Read as", "42")


def test_nothing_to_extract_leaves_the_answer_for_a_person_and_other_steps_still_mark() -> None:
    result = result_of(REASONING_AND_NUMBER_MARKING, Q1="By the formula, no idea")
    assert not result.is_marked
    assert result.marks == 4
    assert "Couldn't find a number." in result.feedback


def test_a_blank_answer_stays_blank() -> None:
    result = result_of(REASONING_AND_NUMBER_MARKING, Q1="")
    assert result.is_marked
    assert result.feedback == "No answer."


@pytest.mark.parametrize(
    ("occurrence", "answer", "correct", "is_marked"),
    [
        ("only", "It is 7", True, True),
        ("only", "7 or 8", False, False),
        ("first", "7 or 8", True, True),
        ("last", "7 or 8", False, True),
    ],
)
def test_occurrence_says_which_match_counts(
    occurrence: str, answer: str, correct: bool, is_marked: bool
) -> None:
    marking = workflow(
        "all_correct",
        {"id": "number", "uses": EXTRACT, "with": {"type": "number", "occurrence": occurrence}},
        {
            "input": "{{ steps.number.output }}",
            "uses": ACCEPTED_ANSWERS,
            "with": {"answers": ["7"]},
        },
        question="Q1",
    )
    result = result_of(marking, Q1=answer)
    assert (result.correct, result.is_marked) == (correct, is_marked)


def extracted(settings: Mapping[str, Any], answer: str, question: TextQuestion) -> str:
    [row] = mark_rows(
        EXTRACT,
        questions=QuestionSet({"Q1": question}),
        question="Q1",
        settings=settings,
        rows=[ExampleRow(answers={"Q1": answer})],
    )
    return row.results["Q1"].feedback


@pytest.mark.parametrize(
    ("pattern", "ignore_case", "feedback"),
    [
        (r"answer: (\w+)", True, "Read as: b"),
        (r"ANSWER: (\w+)", True, "Read as: b"),
        (r"answer: (\w+)", False, "Couldn't find the text."),
        (r"Answer: (\w+)", False, "Read as: B"),
    ],
)
def test_extract_matches_the_answer_ignoring_case_as_its_question_does(
    pattern: str, ignore_case: bool, feedback: str
) -> None:
    question = TextQuestion(reading=TextReading(ignore_case=ignore_case))
    settings = {"type": "text", "pattern": pattern}
    assert extracted(settings, "Working\nAnswer: B", question) == feedback


def test_extract_finds_numbers_in_the_answer_as_its_question_reads_it() -> None:
    question = TextQuestion(reading=TextReading(ignore_punctuation=True))
    assert extracted({}, "It is 2.5", question) == "Read as: 25"


def test_transforms_chain() -> None:
    marking = workflow(
        "all_correct",
        {
            "id": "section",
            "uses": EXTRACT,
            "with": {"type": "text", "pattern": r"Conclusion:\s*(.+)"},
        },
        {
            "id": "last_number",
            "input": "{{ steps.section.output }}",
            "uses": EXTRACT,
            "with": {"type": "number", "occurrence": "last"},
        },
        {**LAST_NUMBER_IN_RANGE, "weight": 1},
        question="Q1",
    )
    assert result_of(marking, Q1="1 2 3\nConclusion: first 7 then 42").correct


def test_weight_zero_decides_but_gives_no_marks() -> None:
    marking = workflow(
        "total",
        {"id": "has_formula", "weight": 0, "uses": KEY_POINTS, "with": {"points": ["formula"]}},
        {
            "if": "{{ steps.has_formula.output.correct }}",
            "uses": KEY_POINTS,
            "weight": 3,
            "with": {"points": ["x"]},
        },
        question="Q1",
    )
    assert result_of(marking, Q1="formula x").marks == 3
    assert result_of(marking, Q1="only x").marks == 0


def test_a_step_reads_a_value_of_the_answer_or_another_answer() -> None:
    marking = workflow(
        "total",
        {
            "input": "{{ values[0] }}",
            "uses": NUMBER_RANGE,
            "with": {"min_value": 1, "max_value": 2},
        },
        {"input": "{{ values[1] }}", "uses": ACCEPTED_ANSWERS, "with": {"answers": ["m"]}},
        question="Q2",
    )
    assert result_of(marking, "Q2", Q2="1.5, m").marks == 2
    assert result_of(marking, "Q2", Q2="9, m").marks == 1
    says_yes = {
        "id": "says_yes",
        "input": "{{ answers.Q1 }}",
        "weight": 0,
        "uses": KEY_POINTS,
        "with": {"points": ["yes"]},
    }
    in_range = {
        "question": "Q3",
        "if": "{{ steps.says_yes.output.correct }}",
        "uses": NUMBER_RANGE,
        "with": {"min_value": 1},
    }
    group = workflow("total", says_yes, in_range)
    assert result_of(group, "Q3", Q1="yes", Q3="5").correct
    assert not result_of(group, "Q3", Q1="no", Q3="5").correct


@pytest.mark.parametrize(
    ("steps", "feedback"),
    [
        ([{"uses": PATTERN, "with": {"pattern": "."}}], "Not The Room cools"),
        ([{"input": "{{ answers.Q4 }}", "uses": PATTERN, "with": {"pattern": "."}}], "the room"),
        (
            [
                {"id": "part", "uses": EXTRACT, "with": {"type": "text", "pattern": "(.+) cools"}},
                {"input": "{{ steps.part.output }}", "uses": PATTERN, "with": {"pattern": "."}},
            ],
            "not the room",
        ),
    ],
)
def test_a_steps_feedback_shows_the_answer_it_read(
    steps: list[dict[str, Any]], feedback: str
) -> None:
    *transforms, check = [{"question": "Q1", **step} for step in steps]
    marking = workflow("all_correct", *transforms, {**check, "feedback": "{{ answer }}"})
    questions = QuestionSet({"Q1": TextQuestion(), "Q4": TextQuestion()})
    [marked] = mark(
        questions,
        MarkScheme.load([marking]),
        [Submission(student_id="s1", answers={"Q1": "Not The Room cools", "Q4": "The Room"})],
    )
    assert marked.results["Q1"].feedback == feedback


def test_a_method_refuses_an_output_of_the_wrong_type() -> None:
    marking = workflow(
        "all_correct",
        {"id": "text", "uses": EXTRACT, "with": {"type": "text", "pattern": "(a)"}},
        {"input": "{{ steps.text.output }}", "uses": NUMBER_RANGE, "with": {"min_value": 1}},
        question="Q1",
    )
    assert problems_of(marking) == ["Number Range can't be used on a text output."]


def test_a_step_cant_read_another_steps_marks_as_input() -> None:
    marking = workflow(
        "all_correct",
        {"id": "key_point", "uses": KEY_POINTS, "with": {"points": ["a"]}},
        {"input": "{{ steps.key_point.output }}", "uses": KEY_POINTS, "with": {"points": ["b"]}},
        question="Q1",
    )
    assert problems_of(marking) == ["Marks can't be read as an answer."]


def test_a_transform_needs_an_id_and_a_later_step_reading_it() -> None:
    transform_without_id = workflow(
        "all_correct",
        {"uses": EXTRACT, "with": {"type": "number"}},
        {"uses": KEY_POINTS, "with": {"points": ["a"]}},
        question="Q1",
    )
    unread_transform = workflow(
        "all_correct",
        EXTRACT_LAST_NUMBER,
        {"uses": KEY_POINTS, "with": {"points": ["a"]}},
        question="Q1",
    )
    assert problems_of(transform_without_id) == ["Give it an ID."]
    assert problems_of(unread_transform) == ["No later step reads Extract."]


def test_a_workflow_needs_a_step_that_gives_marks_and_a_marking_must_give_marks() -> None:
    no_marks = workflow(
        "all_correct", EXTRACT_LAST_NUMBER, {**LAST_NUMBER_IN_RANGE, "weight": 0}, question="Q1"
    )
    assert problems_of(no_marks) == ["Add a step that gives marks."]
    with pytest.raises(InvalidError, match="Only a workflow can use"):
        MarkScheme.load([{"question": "Q1", **EXTRACT_LAST_NUMBER}])


def test_a_pattern_needs_one_group() -> None:
    marking = workflow(
        "all_correct",
        {"id": "text", "uses": EXTRACT, "with": {"type": "text", "pattern": "no groups"}},
        {"input": "{{ steps.text.output }}", "uses": KEY_POINTS, "with": {"points": ["b"]}},
        question="Q1",
    )
    with pytest.raises(InvalidError, match="one group"):
        MarkScheme.load([marking])


def test_a_method_file_with_a_step_output_is_a_transform() -> None:
    methods = BUILTINS.with_files(
        [
            (
                "course/number-in-section",
                """
version: 1
label: Number in Section
question_types: [text]
settings:
  section: {title: Section}
steps:
  - id: part
    uses: gradeflow/extract@1
    with: {type: text, pattern: "{{ settings.section }}"}
  - id: number
    input: "{{ steps.part.output }}"
    uses: gradeflow/extract@1
    with: {type: number, occurrence: last}
output: step_output
output_step: "{{ steps.number.output }}"
""",
            )
        ]
    )
    number_in_section_step = {
        "id": "last_number",
        "uses": "course/number-in-section@1",
        "with": {"section": r"Conclusion:\s*(.+)"},
    }
    marking = workflow(
        "all_correct", number_in_section_step, {**LAST_NUMBER_IN_RANGE, "weight": 1}, question="Q1"
    )
    scheme = MarkScheme.load([marking], methods=methods)
    assert scheme.problems(QUESTIONS) == []
    [marked_submission] = mark(
        QUESTIONS,
        scheme,
        [Submission(student_id="s1", answers={"Q1": "9\nConclusion: 1 then 42"})],
    )
    assert marked_submission.results["Q1"].correct


def test_a_workflow_marking_round_trips_through_load_and_dump() -> None:
    [marking] = MarkScheme.load([REASONING_AND_NUMBER_MARKING]).markings
    assert marking.dump() == REASONING_AND_NUMBER_MARKING


def test_a_step_is_offered_methods_that_fit_what_it_can_read_and_starts_reading_it() -> None:
    draft = MarkingDraft(workflow("total", EXTRACT_LAST_NUMBER, question="Q1"), QUESTIONS, "Q1")
    assert NUMBER_RANGE in {method.ref for method in draft.methods("/with/steps/1")}
    draft.add("/with/steps/1", NUMBER_RANGE)
    assert draft.data["with"]["steps"][1]["input"] == "{{ steps.last_number.output }}"


def test_a_transform_gets_an_id_when_added() -> None:
    draft = MarkingDraft(workflow("all_correct", question="Q1"), QUESTIONS, "Q1")
    draft.add("/with/steps/0", EXTRACT)
    assert draft.data["with"]["steps"][0]["id"] == "extract"


def test_transforms_are_offered_only_among_a_workflows_steps() -> None:
    draft = MarkingDraft(workflow("all_correct", question="Q1"), QUESTIONS, "Q1")
    assert EXTRACT in {method.ref for method in draft.methods("/with/steps/0")}
    assert EXTRACT not in {method.ref for method in draft.methods("")}


def test_without_a_model_sheets_are_marked_as_each_is_read() -> None:
    batch_sizes: list[int] = []

    class BatchSizeRecordingKeyPoint(KeyPoint):
        name = "sample/batch-sizes"
        version = 1
        label = "Batch Sizes"

        def mark_students(
            self,
            settings: KeyPointSettings,
            contexts: Sequence[Context],
            question_id: QuestionId | None,
        ) -> list[QuestionOutcomes]:
            batch_sizes.append(len(contexts))
            return super().mark_students(settings, contexts, question_id)

    methods = Methods([*BUILTINS, BatchSizeRecordingKeyPoint()])
    marking = workflow(
        "all_correct", {"uses": "sample/batch-sizes@1", "with": {"point": "a"}}, question="Q1"
    )
    submissions = [Submission(student_id=str(n), answers={"Q1": f"a {n}"}) for n in range(3)]
    list(mark(QUESTIONS, MarkScheme.load([marking], methods=methods), submissions, chunk_size=3))
    assert batch_sizes == [1, 1, 1]


def test_a_workflow_giving_a_step_output_is_read_by_later_steps() -> None:
    step_output_workflow = {
        "id": "last_number",
        "uses": "gradeflow/workflow@1",
        "with": {
            "output": "step_output",
            "output_step": "{{ steps.number.output }}",
            "steps": [
                {
                    "id": "part",
                    "uses": EXTRACT,
                    "with": {"type": "text", "pattern": r"Conclusion:\s*(.+)"},
                },
                {
                    "id": "number",
                    "input": "{{ steps.part.output }}",
                    "uses": EXTRACT,
                    "with": {"type": "number", "occurrence": "last"},
                },
            ],
        },
    }
    last_number_in_range = {**LAST_NUMBER_IN_RANGE, "weight": 1}
    marking = workflow("all_correct", step_output_workflow, last_number_in_range, question="Q1")
    assert result_of(marking, Q1="9\nConclusion: 1 then 42").correct
    assert not result_of(marking, Q1="9\nConclusion: 1 then 7").correct


@pytest.mark.parametrize(
    ("answer", "feedback", "proportion", "is_marked"),
    [
        ("7 then 42", "Read as: 42", 1.0, True),
        ("It is 2.00", "Read as: 2", 1.0, True),
        ("no number", "Couldn't find a number.", 0.0, False),
        ("", "No answer.", 0.0, True),
    ],
)
def test_testing_a_transform_says_what_it_read(
    answer: str, feedback: str, proportion: float, is_marked: bool
) -> None:
    [row] = mark_rows(
        EXTRACT,
        questions=QUESTIONS,
        question="Q1",
        settings={"occurrence": "last"},
        rows=[ExampleRow(answers={"Q1": answer})],
    )
    result = row.results["Q1"]
    assert (result.feedback, result.proportion, result.is_marked) == (
        feedback,
        proportion,
        is_marked,
    )


SECTION_FILE = """
version: 1
label: Section
question_types: [text]
steps:
  - id: part
    uses: gradeflow/extract@1
    with: {type: text, pattern: "(.+)"}
output: step_output
"""


def test_a_file_giving_a_step_output_must_name_the_step() -> None:
    with pytest.raises(MethodFileError, match="Name the step whose output it gives"):
        BUILTINS.with_files([("course/section", SECTION_FILE)])


def test_marks_by_is_ignored_when_output_is_step_output() -> None:
    methods = BUILTINS.with_files(
        [
            (
                "course/section",
                SECTION_FILE
                + """output_step: "{{ steps.part.output }}"
marks_by: total
""",
            )
        ]
    )
    assert not methods.find("course/section@1").gives_marks({})


def test_only_a_nested_workflow_may_give_a_step_output() -> None:
    nested = workflow("all_correct", EXTRACT_LAST_NUMBER, LAST_NUMBER_IN_RANGE)
    draft = MarkingDraft(workflow("all_correct", nested, question="Q1"), QUESTIONS, "Q1")
    top_level_properties = draft.schema("")["properties"]
    assert not {"output", "output_step"} & top_level_properties.keys()
    nested_properties = draft.schema("/with/steps/0")["properties"]
    assert {"output", "output_step"} <= nested_properties.keys()


def test_output_step_offers_the_workflows_own_transforms_by_label() -> None:
    nested = workflow("all_correct", EXTRACT_LAST_NUMBER, LAST_NUMBER_IN_RANGE)
    draft = MarkingDraft(workflow("all_correct", nested, question="Q1"), QUESTIONS, "Q1")
    field = draft.schema("/with/steps/0")["properties"]["output_step"]
    assert field["enum"] == ["{{ steps.last_number.output }}"]
    assert field[HINT_KEY]["kind"] == "output_step"
    assert field[HINT_KEY]["labels"] == {
        "{{ steps.last_number.output }}": "Extract: Finds the last number."
    }
    assert field[HINT_KEY]["applies_when"] == {"output": ["step_output"]}


def test_a_step_reading_a_transform_says_which_one() -> None:
    first_number = {"id": "first", "uses": EXTRACT, "with": {"occurrence": "first"}}
    named = {**EXTRACT_LAST_NUMBER, "name": "Final number"}
    steps = workflow("all_correct", first_number, named, LAST_NUMBER_IN_RANGE, question="Q1")
    draft = MarkingDraft(steps, QUESTIONS, "Q1")
    assert draft.schema("/with/steps/2")["properties"]["input"][HINT_KEY]["labels"] == {
        "{{ answer }}": "The answer",
        "{{ steps.first.output }}": "Output of Extract: Finds the first number.",
        "{{ steps.last_number.output }}": "Output of Final number",
    }
    assert "Reads the output of Final number" in draft.notes("/with/steps/2")


def test_a_steps_note_numbers_the_step_it_reads_among_all_the_steps_before_it() -> None:
    check = {"id": "a", "name": "Part", "uses": KEY_POINTS, "with": {"points": ["formula"]}}
    named = {**EXTRACT_LAST_NUMBER, "name": "Part"}
    draft = MarkingDraft(
        workflow("all_correct", check, named, LAST_NUMBER_IN_RANGE, question="Q1"), QUESTIONS, "Q1"
    )
    assert "Reads the output of Part 2" in draft.notes("/with/steps/2")


def test_a_steps_note_labels_a_step_whose_output_cant_be_read_yet() -> None:
    reading = {**LAST_NUMBER_IN_RANGE, "question": "Q1"}
    draft = MarkingDraft(workflow("all_correct", EXTRACT_LAST_NUMBER, reading), QUESTIONS, None)
    assert "Reads the output of Extract: Finds the last number." in draft.notes("/with/steps/1")


@pytest.mark.parametrize("steps", [3, True, -1.5])
def test_a_step_output_workflow_whose_steps_arent_a_list_has_a_schema(steps: object) -> None:
    nested = {
        "id": "number",
        "uses": "gradeflow/workflow@1",
        "with": {"output": "step_output", "steps": steps},
    }
    draft = MarkingDraft(workflow("all_correct", nested, question="Q1"), QUESTIONS, "Q1")
    assert draft.schema("/with/steps/0")["properties"]["output_step"]["enum"] == []


@pytest.mark.parametrize("input_", [["a"], 3])
def test_a_step_whose_input_isnt_text_has_a_schema(input_: object) -> None:
    check = {"input": input_, "uses": NUMBER_RANGE, "with": {"min_value": 1}}
    draft = MarkingDraft(workflow("all_correct", check, question="Q2"), QUESTIONS, "Q2")
    assert draft.schema("/with/steps/0")["properties"]["input"]["enum"] == [
        "{{ answer }}",
        "{{ values[0] }}",
    ]
    assert draft.settings_holding_steps("/with/steps/0") == []


def test_transforms_with_the_same_description_are_numbered() -> None:
    other = {"id": "other", "uses": EXTRACT, "with": {"occurrence": "last"}}
    draft = MarkingDraft(
        workflow("all_correct", EXTRACT_LAST_NUMBER, other, LAST_NUMBER_IN_RANGE, question="Q1"),
        QUESTIONS,
        "Q1",
    )
    labels = draft.schema("/with/steps/2")["properties"]["input"][HINT_KEY]["labels"]
    assert labels["{{ steps.last_number.output }}"] == "Output of Extract: Finds the last number. 1"
    assert labels["{{ steps.other.output }}"] == "Output of Extract: Finds the last number. 2"


def test_a_method_draft_chooses_marks_or_a_step_output() -> None:
    draft = MethodDraft.new(QUESTIONS, name="m", version=1, label="M", summary="M.")
    top_level_properties = draft.steps_draft.schema("")["properties"]
    assert {"output", "marks_by", "output_step"} <= top_level_properties.keys()
    assert top_level_properties["marks_by"][HINT_KEY]["applies_when"] == {"output": ["marks"]}
    assert top_level_properties["output_step"][HINT_KEY]["applies_when"] == {
        "output": ["step_output"]
    }
    assert draft.steps_draft.position("").is_in_workflow


UNIT_PART = {
    "id": "part",
    "input": "{{ values[1] }}",
    "uses": EXTRACT,
    "with": {"type": "text", "pattern": "(.+)"},
}


def test_a_transform_reading_a_value_gives_what_that_value_is() -> None:
    read_part = {"input": "{{ steps.part.output }}", "uses": PATTERN, "with": {"pattern": "m"}}
    marking = workflow("all_correct", UNIT_PART, read_part, question="Q2")
    assert problems_of(marking) == []
    assert result_of(marking, "Q2", Q2="1.5, m").correct


def test_a_workflow_giving_a_step_output_gives_what_its_step_read() -> None:
    step_output_workflow = {
        "id": "unit",
        "uses": "gradeflow/workflow@1",
        "with": {
            "output": "step_output",
            "output_step": "{{ steps.part.output }}",
            "steps": [UNIT_PART],
        },
    }
    read_unit = {"input": "{{ steps.unit.output }}", "uses": PATTERN, "with": {"pattern": "m"}}
    marking = workflow("all_correct", step_output_workflow, read_unit, question="Q2")
    assert problems_of(marking) == []
    assert result_of(marking, "Q2", Q2="1.5, m").correct


@pytest.mark.parametrize(
    ("settings", "problems"),
    [
        ({"kind": "text"}, []),
        ({}, []),
        ({"kind": "number"}, ["Pattern can't be used on a number output."]),
    ],
)
def test_a_composed_transform_gives_the_type_its_settings_choose(
    settings: dict[str, str], problems: list[str]
) -> None:
    methods = BUILTINS.with_files(
        [
            (
                "course/find",
                """
version: 1
label: Find
question_types: [text]
settings:
  kind: {title: Kind, default: text}
steps:
  - id: part
    uses: gradeflow/extract@1
    with: {type: "{{ settings.kind }}", pattern: "(.+)"}
output: step_output
output_step: "{{ steps.part.output }}"
""",
            )
        ]
    )
    marking = workflow(
        "all_correct",
        {"id": "found", "uses": "course/find@1", "with": settings},
        {"input": "{{ steps.found.output }}", "uses": PATTERN, "with": {"pattern": "a"}},
        question="Q1",
    )
    found = MarkScheme.load([marking], methods=methods).problems(QUESTIONS)
    assert [problem.message for problem in found] == problems


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("1,000 apples", 1000), ("about -1,234.5 kg", -1234.5), ("1,2", None)],
)
def test_extract_reads_thousands_separators(answer: str, expected: float | None) -> None:
    extract = {"id": "number", "uses": EXTRACT}
    in_range = {
        "input": "{{ steps.number.output }}",
        "uses": NUMBER_RANGE,
        "with": {"min_value": expected or 0, "max_value": expected or 0},
    }
    result = result_of(workflow("all_correct", extract, in_range, question="Q1"), Q1=answer)
    assert result.is_marked is (expected is not None)
    assert result.correct is (expected is not None)


def test_extract_finds_no_text_in_an_empty_group() -> None:
    extract = {
        "id": "part",
        "uses": EXTRACT,
        "with": {"type": "text", "pattern": r"Answer:\s*(.*)"},
    }
    read_part = {"input": "{{ steps.part.output }}", "uses": PATTERN, "with": {"pattern": "a"}}
    result = result_of(workflow("all_correct", extract, read_part, question="Q1"), Q1="Answer: ")
    assert not result.is_marked
    assert "Couldn't find the text." in result.feedback


def test_a_step_reads_an_answer_whose_question_id_is_not_a_name() -> None:
    questions = QuestionSet({"Q-1": TextQuestion(), "Q2": TextQuestion()})
    key_point = {"question": "Q2", "uses": KEY_POINTS, "with": {"points": ["yes"]}}
    draft = MarkingDraft(workflow("all_correct", key_point), questions, None)
    offered = draft.schema("/with/steps/0")["properties"]["input"]["enum"]
    assert "{{ answers['Q-1'] }}" in offered
    assert "{{ answers.Q2 }}" in offered
    marking = workflow("all_correct", {**key_point, "input": "{{ answers['Q-1'] }}"})
    scheme = MarkScheme.load([marking])
    assert scheme.problems(questions) == []
    [marked_submission] = mark(
        questions, scheme, [Submission(student_id="s1", answers={"Q-1": "yes"})]
    )
    assert marked_submission.results["Q2"].correct


def test_a_step_of_a_group_reading_an_output_must_name_its_question() -> None:
    marking = {
        "uses": "gradeflow/workflow@1",
        "with": {
            "steps": [
                {"id": "number", "question": "Q1", "uses": EXTRACT},
                {
                    "input": "{{ steps.number.output }}",
                    "uses": NUMBER_RANGE,
                    "with": {"min_value": 1, "max_value": 2},
                },
            ],
        },
    }
    assert [
        (problem.message, problem.field)
        for problem in MarkScheme.load([marking]).problems(QUESTIONS)
    ] == [("Choose the question it checks.", ("question",))]


def test_a_workflow_changed_to_give_a_steps_output_gets_an_id() -> None:
    nested = workflow("all_correct", EXTRACT_LAST_NUMBER, LAST_NUMBER_IN_RANGE)
    draft = MarkingDraft(workflow("all_correct", nested, question="Q1"), QUESTIONS, "Q1")
    draft.update(
        "/with/steps/0",
        {"output": "step_output", "output_step": "{{ steps.last_number.output }}"},
    )
    assert draft.data["with"]["steps"][0]["id"]


def test_a_method_giving_a_steps_output_has_no_id_of_its_own() -> None:
    draft = MethodDraft.new(QUESTIONS, name="m", version=1, label="M", summary="M.")
    draft.steps_draft.update("", {"output": "step_output"})
    assert "id" not in draft.steps_draft.data


def test_extracted_text_that_is_the_blank_marker_is_blank() -> None:
    settings = {"type": "text", "pattern": r"Answer: (.+)"}
    assert extracted(settings, "Answer: N/A", TextQuestion()) == "No answer."


@pytest.mark.parametrize(
    ("inner_step", "output_step"),
    [
        ({**EXTRACT_LAST_NUMBER, "input": "nonsense"}, "{{ steps.last_number.output }}"),
        (EXTRACT_LAST_NUMBER, "nonsense"),
    ],
)
def test_a_draft_reports_a_malformed_input_or_output_step_of_a_step_output_workflow(
    inner_step: dict[str, Any], output_step: str
) -> None:
    nested = {
        "id": "number",
        "uses": "gradeflow/workflow@1",
        "with": {"output": "step_output", "output_step": output_step, "steps": [inner_step]},
    }
    reading = {**LAST_NUMBER_IN_RANGE, "input": "{{ steps.number.output }}"}
    draft = MarkingDraft(workflow("all_correct", nested, reading, question="Q1"), QUESTIONS, "Q1")
    assert "/with/steps/1" in draft.paths()
    assert draft.problems


@pytest.mark.parametrize(
    ("question", "problems"),
    [
        (["Q1"], ["question: Must be text."]),
        (
            "Q9",
            [
                "question: Q9 isn't one of the questions.",
                "Use a group marking to read or award Q9.",
            ],
        ),
    ],
)
def test_a_draft_reports_only_the_bad_question_of_a_step_in_a_step_output_workflow(
    question: object, problems: list[str]
) -> None:
    nested = {
        "id": "number",
        "uses": "gradeflow/workflow@1",
        "with": {
            "output": "step_output",
            "output_step": "{{ steps.last_number.output }}",
            "steps": [{**EXTRACT_LAST_NUMBER, "question": question}],
        },
    }
    reading = {**LAST_NUMBER_IN_RANGE, "input": "{{ steps.number.output }}"}
    draft = MarkingDraft(workflow("total", nested, reading, question="Q1"), QUESTIONS, "Q1")
    assert [str(problem) for problem in draft.problems] == problems


@pytest.mark.parametrize("reads_q1", [{"input": "{{ answers.Q1 }}"}, {"question": "Q1"}])
def test_a_workflow_giving_a_step_output_gives_what_its_step_read_from_another_question(
    reads_q1: dict[str, str],
) -> None:
    part = {"id": "part", **reads_q1, "uses": EXTRACT, "with": {"type": "text", "pattern": "(.+)"}}
    step_output_workflow = {
        "id": "found",
        "question": "Q2",
        "uses": "gradeflow/workflow@1",
        "with": {
            "output": "step_output",
            "output_step": "{{ steps.part.output }}",
            "steps": [part],
        },
    }
    read_found = {
        "question": "Q2",
        "input": "{{ steps.found.output }}",
        "uses": PATTERN,
        "with": {"pattern": "m"},
    }
    marking = workflow("all_correct", step_output_workflow, read_found)
    part_step = valid_step(part, BUILTINS.by_ref)
    [result] = transform(part_step, [Context({"Q1": "m"}, QUESTIONS, {"Q1": "m"})], "Q2")
    assert result.question is not None
    draft = MarkingDraft(marking, QUESTIONS, None)
    assert draft.position("/with/steps/1").question_type == result.question.type == "text"
    assert result_of(marking, "Q2", Q1="m", Q2="1.5, m").correct


def test_a_step_output_workflow_whose_output_step_names_an_answer_gives_what_it_reads() -> None:
    nested = {
        "id": "number",
        "uses": "gradeflow/workflow@1",
        "with": {
            "output": "step_output",
            "output_step": "{{ answers.Q3 }}",
            "steps": [EXTRACT_LAST_NUMBER],
        },
    }
    reading = {**LAST_NUMBER_IN_RANGE, "input": "{{ steps.number.output }}"}
    draft = MarkingDraft(workflow("total", nested, reading, question="Q1"), QUESTIONS, "Q1")
    assert draft.position("/with/steps/1").question_type == "text"


def test_a_workflow_cant_give_a_checks_marks_as_its_output() -> None:
    check = {"id": "point", "uses": KEY_POINTS, "with": {"points": ["formula"]}}
    nested = {
        "id": "found",
        "uses": "gradeflow/workflow@1",
        "with": {
            "output": "step_output",
            "output_step": "{{ steps.point.output }}",
            "steps": [check],
        },
    }
    reading = {**LAST_NUMBER_IN_RANGE, "input": "{{ steps.found.output }}"}
    problems = MarkScheme.load([workflow("all_correct", nested, reading, question="Q1")]).problems(
        QUESTIONS
    )
    assert (
        "Marks can't be read as an answer.",
        "/markings/0/with/steps/0/output_step",
    ) in [(problem.message, problem.where) for problem in problems]


def test_until_a_transforms_settings_are_valid_it_is_taken_to_give_what_it_reads() -> None:
    def type_read_after(extract: Mapping[str, Any]) -> str | None:
        reading = {**LAST_NUMBER_IN_RANGE, "input": "{{ steps.last_number.output }}"}
        draft = MarkingDraft(workflow("total", extract, reading, question="Q1"), QUESTIONS, "Q1")
        return draft.position("/with/steps/1").question_type

    assert type_read_after(EXTRACT_LAST_NUMBER) == "number"
    assert type_read_after({**EXTRACT_LAST_NUMBER, "with": "x"}) == QUESTIONS["Q1"].type == "text"
