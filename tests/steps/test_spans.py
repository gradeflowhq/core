from collections.abc import Mapping
from typing import Any

import pytest

from gradeflow_core import (
    BUILTINS,
    ListQuestion,
    MarkScheme,
    Methods,
    QuestionResult,
    QuestionSet,
    Span,
    Submission,
    TextQuestion,
    mark,
)
from gradeflow_core.editing import ExampleRow, mark_rows
from gradeflow_core.questions.parts import AnswerPart

from ..methods.helpers import multipart, workflow

EXTRACT = "gradeflow/extract@1"
QUESTIONS = QuestionSet(
    {
        "Q1": TextQuestion(max_marks=2),
        "Q2": multipart("number", "text"),
        "Q3": TextQuestion(),
        "Q4": ListQuestion(
            item="parts", parts=[AnswerPart(name="ingredient"), AnswerPart(name="amount")]
        ),
    }
)
LAST_NUMBER = {"id": "number", "uses": EXTRACT, "with": {"occurrence": "last"}}
SECTION = {
    "id": "section",
    "uses": EXTRACT,
    "with": {"type": "text", "pattern": r"conclusion:(.*)"},
}
NUMBER_IN_SECTION = {**LAST_NUMBER, "input": "{{ steps.section.output }}"}
IS_42 = {
    "input": "{{ steps.number.output }}",
    "uses": "gradeflow/number-range@1",
    "with": {"min_value": 42, "max_value": 42},
}
CONCLUSION = "9\nConclusion: 1 then 42"
"""Read by its question as ``9\\nconclusion: 1 then 42``, with 42 at characters 21 up to, not
including, 23."""


def spans_of(
    marking: Mapping[str, Any],
    question_id: str = "Q1",
    methods: Methods = BUILTINS,
    **answers: str,
) -> tuple[Span, ...]:
    scheme = MarkScheme.load([marking], methods=methods)
    assert scheme.problems(QUESTIONS) == []
    [marked] = mark(QUESTIONS, scheme, [Submission(student_id="s1", answers=answers)])
    return marked.results[question_id].spans


def test_a_check_reading_a_transform_gets_the_span_it_extracted() -> None:
    marking = workflow("all_correct", LAST_NUMBER, IS_42, question="Q1")
    assert spans_of(marking, Q1="By the formula, x = 42") == (Span(20, 22),)


def test_a_text_span_leaves_out_the_spaces_its_question_trims() -> None:
    text = {"id": "text", "uses": EXTRACT, "with": {"type": "text", "pattern": r"answer:(.*)"}}
    is_b = {
        "input": "{{ steps.text.output }}",
        "uses": "gradeflow/accepted-answers@1",
        "with": {"answers": ["b"]},
    }
    assert spans_of(workflow("all_correct", text, is_b, question="Q1"), Q1="Answer:  B ") == (
        Span(9, 10),
    )


def test_chained_transforms_map_spans_to_the_answer() -> None:
    marking = workflow("all_correct", SECTION, NUMBER_IN_SECTION, IS_42, question="Q1")
    assert spans_of(marking, Q1=CONCLUSION) == (Span(21, 23),)


def test_a_workflow_giving_a_step_output_maps_spans_to_the_answer() -> None:
    number_workflow = {
        "id": "number",
        "uses": "gradeflow/workflow@1",
        "with": {
            "output": "step_output",
            "output_step": "{{ steps.number.output }}",
            "steps": [SECTION, NUMBER_IN_SECTION],
        },
    }
    marking = workflow("all_correct", number_workflow, IS_42, question="Q1")
    assert spans_of(marking, Q1=CONCLUSION) == (Span(21, 23),)


def test_a_method_file_giving_a_step_output_maps_spans_to_the_answer() -> None:
    methods = BUILTINS.with_files(
        [
            (
                "course/number-in-section",
                """
version: 1
label: Number in Section
question_types: [text]
steps:
  - id: section
    uses: gradeflow/extract@1
    with: {type: text, pattern: "conclusion:(.*)"}
  - id: number
    input: "{{ steps.section.output }}"
    uses: gradeflow/extract@1
    with: {type: number, occurrence: last}
output: step_output
output_step: "{{ steps.number.output }}"
""",
            )
        ]
    )
    step = {"id": "number", "uses": "course/number-in-section@1"}
    marking = workflow("all_correct", step, IS_42, question="Q1")
    assert spans_of(marking, methods=methods, Q1=CONCLUSION) == (Span(21, 23),)


def test_a_span_in_a_part_counts_from_that_parts_start() -> None:
    number = {**LAST_NUMBER, "input": "{{ answer.part_2 }}"}
    marking = workflow("all_correct", number, {**IS_42, "weight": 1}, question="Q2")
    assert spans_of(marking, "Q2", Q2="3, About 42 kg") == (Span(6, 8, path="/part_2"),)


def test_a_span_in_a_part_of_an_item_is_at_that_items_part() -> None:
    check = workflow("all_correct", {**LAST_NUMBER, "input": "{{ answer.amount }}"}, IS_42)
    marking = {"question": "Q4", "uses": "gradeflow/item-by-item@1", "with": {"checks": [check]}}
    answer = "water, 2 l, salt, about 42 g"
    assert spans_of(marking, "Q4", Q4=answer) == (Span(6, 8, path="/items/1/amount"),)


def test_reading_another_questions_answer_gives_no_spans() -> None:
    number = {**LAST_NUMBER, "question": "Q3", "input": "{{ answers.Q1 }}"}
    marking = workflow("total", number, {**IS_42, "question": "Q3"})
    assert spans_of(marking, "Q3", Q1="It is 42", Q3="Forty-two") == ()


def test_a_check_marking_a_different_question_from_its_transform_gets_no_spans() -> None:
    marking = workflow("total", {**LAST_NUMBER, "question": "Q1"}, {**IS_42, "question": "Q3"})
    assert spans_of(marking, "Q3", Q1="It is 42", Q3="Forty-two") == ()


@pytest.mark.parametrize("answer", ["No number", "7 or 42"])
def test_a_failed_extract_gives_no_spans(answer: str) -> None:
    only_number = {"id": "number", "uses": EXTRACT}
    assert spans_of(workflow("all_correct", only_number, IS_42, question="Q1"), Q1=answer) == ()


def test_spans_that_overlap_are_merged() -> None:
    text = {"id": "text", "uses": EXTRACT, "with": {"type": "text", "pattern": r"answer:(.*)"}}
    has_kg = {"input": "{{ steps.text.output }}", "uses": "gradeflow/pattern@1"}
    marking = workflow(
        "total", text, {**has_kg, "with": {"pattern": "kg"}}, LAST_NUMBER, IS_42, question="Q1"
    )
    assert spans_of(marking, Q1="Answer: 42 kg") == (Span(8, 13),)


def test_marking_rows_with_a_transform_gives_its_spans() -> None:
    [row] = mark_rows(
        EXTRACT,
        questions=QUESTIONS,
        question="Q1",
        settings={},
        rows=[ExampleRow(answers={"Q1": "It is 42"})],
    )
    assert row.results["Q1"].spans == (Span(6, 8),)


def test_a_results_spans_are_sorted_with_overlapping_and_empty_ones_merged_away() -> None:
    spans = (Span(3, 6), Span(5, 5), Span(0, 4), Span(1, 2, path="/part_2"))
    result = QuestionResult(proportion=1, max_marks=1, spans=spans)
    assert result.spans == (Span(0, 6), Span(1, 2, path="/part_2"))


@pytest.mark.parametrize(("start", "end"), [(-1, 2), (3, 2)])
def test_a_span_rejects_a_negative_start_or_an_end_before_its_start(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="0 ≤ start ≤ end"):
        Span(start, end)
