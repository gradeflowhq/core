import json
import os
import re
import subprocess
import sys
from typing import Any
from unittest.mock import Mock

import pytest

from gradeflow_core import (
    BUILTINS,
    GradeFlowError,
    Method,
    Methods,
    QuestionSet,
    mark_examples,
)
from gradeflow_core.editing import ExpectedResult, RowResult, mark_rows
from tests.fake_models import (
    FAST_AND_CAREFUL_ENDPOINT,
    InstallFakeModels,
    example_verdicts,
)

BASIC_METHOD_NAMES = {
    "gradeflow/accepted-answers",
    "gradeflow/key-points",
    "gradeflow/answer-criteria",
    "gradeflow/equivalent-expression",
    "gradeflow/number-range",
    "gradeflow/multiple-choice",
    "gradeflow/multiple-values",
    "gradeflow/order",
    "gradeflow/length",
    "gradeflow/full-marks",
}
GUIDE_PATTERN = re.compile(
    r"^\*\*Use it when:\*\* [A-Z].+"
    r"\n\n\*\*How it marks:\*\* [A-Z*].+"
    r"(\n\n\*\*Good to know:\*\* [A-Z*].+)?"
    r"(\n\n\*\*Related:\*\* [A-Z*].+)?$"
)
SUMMARY_OPENINGS = ("Full marks ", "Marks ", "Several checks ", "Your Python code ")
PUBLIC_METHODS = [method for method in BUILTINS if method.level != "internal"]
BY_NAME = {method.name: method for method in BUILTINS}


@pytest.mark.parametrize("method", PUBLIC_METHODS, ids=lambda method: method.name)
def test_every_method_explains_itself_plainly(method: Method[Any]) -> None:
    assert method.summary.endswith(".")
    assert ". " not in method.summary
    assert len(method.summary.split()) <= 10, method.summary
    assert method.summary.startswith(SUMMARY_OPENINGS), method.summary
    assert GUIDE_PATTERN.match(method.guide), method.guide
    assert method.level == ("basic" if method.name in BASIC_METHOD_NAMES else "advanced")
    assert bool(method.question_types) or method.marks_question_group
    assert len(method.examples) >= 2


def test_method_names_are_their_labels() -> None:
    assert all(
        method.name == f"gradeflow/{method.label.lower().replace(' ', '-')}" for method in BUILTINS
    )
    labels = [method.label for method in BUILTINS.latest()]
    assert len(labels) == len(set(labels))


def test_example_feedback_is_never_empty() -> None:
    assert all(
        row.expected and all(expected.feedback for expected in row.expected.values())
        for method in BUILTINS
        for example in method.examples
        for row in example.rows
    )


EXAMPLE_ROW_MARKS = {
    "gradeflow/accepted-answers": [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0]],
    "gradeflow/key-points": [[3, 2, 1], [1, 1, 0], [2, 1, 0]],
    "gradeflow/answer-criteria": [[3, 0, 2], [2, 1, 0], [2, 1, 0]],
    "gradeflow/equivalent-expression": [[1, 1, 0, 0], [1, 0, 0]],
    "gradeflow/number-range": [[1, 1, 0, 0], [1, 0]],
    "gradeflow/multiple-choice": [[1, 0, 0], [2, 1, 1, 0]],
    "gradeflow/multiple-values": [[2, 1, 0], [2, 2, 1]],
    "gradeflow/order": [[4, 3, 3, 1], [1, 0]],
    "gradeflow/length": [[1, 1, 0, 0], [1, 0]],
    "gradeflow/full-marks": [[1, 1, 1], [2, 2]],
    "gradeflow/combination": [[1, 0, 0], [2, 1, 0]],
    "gradeflow/not-allowed": [[1, 0], [1, 0]],
    "gradeflow/pattern": [[1, 0, 0], [1, 1, 0]],
    "gradeflow/code-tests": [[2, 1], [1, 0]],
    "gradeflow/code-uses": [[1, 0], [1, 0]],
    "gradeflow/custom-code": [[1, 0, 0], [1, 0]],
    "gradeflow/conditional": [[1, 1, 0], [1, 1, 0]],
    "gradeflow/answer-sets": [[2, 1, 1, 0], [2, 2, 1]],
    "gradeflow/formula-of-answers": [[1, 1, 0, 0], [1, 1, 0]],
    "gradeflow/custom-code-for-a-group": [[2, 1, 1], [2, 1, 1]],
}


@pytest.mark.parametrize(("name", "marks"), EXAMPLE_ROW_MARKS.items())
def test_methods_declare_expected_results(name: str, marks: list[list[float]]) -> None:
    method = BY_NAME[name]
    assert [
        [sum(expected.marks for expected in row.expected.values()) for row in example.rows]
        for example in method.examples
    ] == marks
    for example in method.examples:
        assert example.title
        assert example.description
        assert all(
            row.expected
            and all(
                0 <= expected.marks <= example.questions[key].max_marks
                for key, expected in row.expected.items()
            )
            for row in example.rows
        )


def test_every_public_method_has_its_marks_pinned() -> None:
    assert {method.name for method in PUBLIC_METHODS} == set(EXAMPLE_ROW_MARKS)


def test_examples_across_questions_declare_each_questions_result() -> None:
    follow_through_row = BY_NAME["gradeflow/answer-sets"].examples[0].rows[1]
    assert follow_through_row.answers == {"Q1": "10", "Q2": "6.25"}
    half = ExpectedResult(marks=0.5, feedback="Perimeter as 6 + 4 (50% of the marks).")
    assert follow_through_row.expected == {"Q1": half, "Q2": half}


def test_a_row_matches_only_when_it_declares_every_question_marked() -> None:
    method = BY_NAME["gradeflow/answer-sets"]
    example = method.examples[0]
    row = example.rows[0]
    row_without_q2 = row.model_copy(update={"expected": {"Q1": row.expected["Q1"]}})
    full_row, partial_row = mark_rows(
        method.ref,
        questions=QuestionSet(example.questions),
        question=None,
        inputs=example.with_,
        rows=[row, row_without_q2],
    )
    assert (full_row.matches_expected, partial_row.matches_expected) == (True, False)


@pytest.mark.parametrize("seed", ["1", "2", "3"])
def test_example_results_follow_question_order_across_hash_seeds(seed: str) -> None:
    code = (
        "import json; from gradeflow_core import BUILTINS, mark_examples; "
        "print(json.dumps([[result.feedback for result in row.results.values()] for row in "
        "mark_examples('gradeflow/answer-sets@1')[0].rows]))"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}, text=True
    )
    assert json.loads(output) == [
        ["Correct.", "Correct."],
        ["Perimeter as 6 + 4 (50% of the marks).", "Perimeter as 6 + 4 (50% of the marks)."],
        ["Correct.", "Expected 25."],
        ["Expected 20.", "Expected 25."],
    ]


def test_evaluation_refuses_unknown_methods() -> None:
    with pytest.raises(GradeFlowError, match=re.escape("There's no method gradeflow/nope@1.")):
        mark_examples("gradeflow/nope@1")
    with pytest.raises(GradeFlowError, match=re.escape("There's no method gradeflow/key-point.")):
        mark_examples("gradeflow/key-point")


EXAMPLE_METHOD_FILE = """label: Example
version: 1
summary: Executes code.
question_types: [text]
checks:
  - uses: gradeflow/custom-code@1
    with:
      code: |
        proportion = 1
        feedback = 'Worked.'
examples:
  - title: A sample
    description: A declared description.
    questions: {Q1: {type: text}}
    rows:
      - answers: {Q1: sample}
        expected:
          Q1: {marks: 0, feedback: A declared expectation.}
"""


def test_methods_hold_declared_content_without_marking(monkeypatch: pytest.MonkeyPatch) -> None:
    methods = BUILTINS.with_files([("sample/example", EXAMPLE_METHOD_FILE)])
    mark = Mock()
    monkeypatch.setattr("gradeflow_core.marking.examples.mark", mark)
    [example] = methods[-1].examples
    assert example.description == "A declared description."
    assert example.rows[0].answers == {"Q1": "sample"}
    assert example.rows[0].expected == {
        "Q1": ExpectedResult(marks=0, feedback="A declared expectation.")
    }
    assert BY_NAME["gradeflow/key-points"].examples
    mark.assert_not_called()


def test_explicit_evaluation_leaves_declared_expectations_unchanged() -> None:
    methods = BUILTINS.with_files([("sample/example", EXAMPLE_METHOD_FILE)])
    declared = [example.model_dump() for example in methods[-1].examples]
    [row] = mark_examples("sample/example@1", methods=methods)[0].rows
    assert (row.total_marks, row.results["Q1"].feedback, row.matches_expected) == (
        1,
        "Worked.",
        False,
    )
    assert [example.model_dump() for example in methods[-1].examples] == declared
    assert len(methods) == len(BUILTINS) + 1


def test_example_evaluation_keeps_rows_whose_code_failed() -> None:
    methods = BUILTINS.with_files(
        [
            (
                "sample/example",
                EXAMPLE_METHOD_FILE.replace("proportion = 1", "raise ValueError('Broken')"),
            )
        ]
    )
    [row] = mark_examples("sample/example@1", methods=methods)[0].rows
    result = row.results["Q1"]
    assert (result.is_marked, row.matches_expected, row.total_marks) == (False, False, 0)
    assert "Broken" in result.feedback


def test_example_rows_must_answer_the_questions() -> None:
    methods = BUILTINS.with_files(
        [("sample/example", EXAMPLE_METHOD_FILE.replace("{Q1: sample}", "{Q2: sample}"))]
    )
    with pytest.raises(GradeFlowError, match=re.escape("A sample: Row 1 has no answer to Q1.")):
        mark_examples("sample/example@1", methods=methods)


def test_example_evaluation_checks_question_contract_and_supports_one_question_groups() -> None:
    number_question_file = EXAMPLE_METHOD_FILE.replace("type: text", "type: number")
    methods = BUILTINS.with_files([("sample/example", number_question_file)])
    with pytest.raises(GradeFlowError, match=r"^A sample: "):
        mark_examples("sample/example@1", methods=methods)

    group_file = EXAMPLE_METHOD_FILE.replace(
        "question_types: [text]", "marks_question_group: true"
    ).replace(
        "- uses: gradeflow/custom-code@1", "- question: Q1\n    uses: gradeflow/custom-code@1"
    )
    methods = BUILTINS.with_files([("sample/example", group_file)])
    assert mark_examples("sample/example@1", methods=methods)[0].rows[0].total_marks == 1


def test_actual_rows_are_independent_from_declared_rows() -> None:
    method = BY_NAME["gradeflow/pattern"]
    declared = [example.model_dump() for example in method.examples]
    actual = mark_examples(method.ref)
    actual[0].rows[0].results.clear()
    actual[0].rows[0].answers["Q1"] = "Changed"
    assert [example.model_dump() for example in method.examples] == declared


MEANING_MODEL_METHODS = {"gradeflow/accepted-answers", "gradeflow/key-points"}
"""Methods whose examples run the meaning model."""


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(
            method,
            marks=pytest.mark.ml if method.name in MEANING_MODEL_METHODS else (),
            id=method.ref,
        )
        for method in BUILTINS
    ],
)
def test_built_in_examples_declare_what_evaluating_them_gives(
    method: Method[Any], fake_models: InstallFakeModels
) -> None:
    fake_models(example_verdicts)
    actual = mark_examples(method.ref, model_endpoint=FAST_AND_CAREFUL_ENDPOINT)
    assert [example.description for example in actual] == [
        example.description for example in method.examples
    ]
    assert [[as_expected(row) for row in example.rows] for example in actual] == [
        [row.expected for row in example.rows] for example in method.examples
    ]
    assert all(row.matches_expected for example in actual for row in example.rows)


def as_expected(row: RowResult) -> dict[str, ExpectedResult]:
    return row.as_example().expected


def test_examples_use_supplied_nested_methods_without_fallback() -> None:
    methods = BUILTINS.with_files([("sample/example", EXAMPLE_METHOD_FILE)])
    only_example = Methods(method for method in methods if method.name == "sample/example")
    with pytest.raises(GradeFlowError, match=r"gradeflow/(custom-code|combination)@1"):
        mark_examples("sample/example@1", methods=only_example)


def test_examples_without_expectations_can_be_read_and_explicitly_evaluated() -> None:
    without_expectations = EXAMPLE_METHOD_FILE.split("        expected:")[0]
    methods = BUILTINS.with_files([("sample/example", without_expectations)])
    declared = methods[-1].examples[0].rows[0]
    assert declared.answers == {"Q1": "sample"}
    assert declared.expected == {}
    actual = mark_examples("sample/example@1", methods=methods)[0].rows[0]
    assert (actual.total_marks, actual.results["Q1"].feedback, actual.matches_expected) == (
        1,
        "Worked.",
        None,
    )
    assert declared.expected == {}
