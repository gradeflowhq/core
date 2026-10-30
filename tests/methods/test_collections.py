import re

import pytest

from gradeflow_core import (
    BUILTINS,
    Check,
    GradeFlowError,
    InvalidError,
    MarkScheme,
    MethodFileError,
    Methods,
    Question,
    QuestionSet,
    Submission,
    mark,
)
from gradeflow_core.editing import methods_used, rewrite_references
from gradeflow_core.methods.logic.full_marks import FullMarks
from gradeflow_core.methods.outcome import Outcome
from gradeflow_core.methods.settings import Settings
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.types import Answer


def definition(reference: str) -> str:
    return (
        "label: Wrapper\nversion: 1\nsummary: A wrapper.\nquestion_types: [text]\n"
        f"checks:\n  - uses: {reference}\n"
    )


def test_methods_find_a_method_by_its_ref_and_slice_as_methods() -> None:
    assert BUILTINS.find("gradeflow/key-points@1") is BUILTINS.by_ref["gradeflow/key-points@1"]
    assert repr(BUILTINS.find("gradeflow/key-points@1")) == "<Method gradeflow/key-points@1>"
    with pytest.raises(GradeFlowError, match="There's no method gradeflow/key-points@9"):
        BUILTINS.find("gradeflow/key-points@9")
    assert isinstance(BUILTINS[1:], Methods)
    assert list(BUILTINS[1:]) == list(BUILTINS)[1:]
    assert len(Methods()) == 0


def test_a_check_uses_the_built_in_methods_unless_given_others() -> None:
    full_marks = BUILTINS.find("gradeflow/full-marks@1")
    assert Check.load({"uses": full_marks.ref}).method is full_marks


def test_loading_preserves_input_order_and_does_not_modify_supplied_collections() -> None:
    original = list(BUILTINS)
    files = [
        ("sample/outer", definition("sample/inner@1")),
        ("sample/inner", definition("gradeflow/full-marks@1")),
    ]
    methods = BUILTINS.with_files(files)
    assert list(BUILTINS) == original
    assert list(methods[:-2]) == original
    assert [method.ref for method in methods[-2:]] == ["sample/outer@1", "sample/inner@1"]
    assert methods.dependencies("sample/outer@1") == {
        "sample/inner@1",
        "gradeflow/combination@1",
        "gradeflow/full-marks@1",
    }


def versioned(version: int, reference: str) -> str:
    return definition(reference).replace("version: 1", f"version: {version}")


def test_versions_of_a_method_are_distinct_and_latest_keeps_the_highest() -> None:
    files = [
        ("sample/wrapper", versioned(2, "gradeflow/full-marks@1")),
        ("sample/other", definition("gradeflow/full-marks@1")),
        ("sample/wrapper", versioned(1, "gradeflow/not-allowed@1")),
    ]
    methods = BUILTINS.with_files(files)
    loaded = methods[len(BUILTINS) :]
    assert [method.ref for method in loaded] == [
        "sample/wrapper@2",
        "sample/other@1",
        "sample/wrapper@1",
    ]
    assert {method.name for method in loaded} == {"sample/wrapper", "sample/other"}
    assert Check.load({"uses": "sample/wrapper@1"}, methods=methods).method is loaded[2]
    assert Check.load({"uses": "sample/wrapper@2"}, methods=methods).method is loaded[0]
    with pytest.raises(InvalidError, match="There's no method sample/wrapper"):
        Check.load({"uses": "sample/wrapper"}, methods=methods)
    assert [method.ref for method in loaded.latest()] == ["sample/wrapper@2", "sample/other@1"]
    assert list(BUILTINS.latest()) == list(BUILTINS)


def test_methods_are_equal_when_they_hold_the_same_methods_in_order() -> None:
    assert Methods(BUILTINS) == BUILTINS
    assert BUILTINS[:2] != BUILTINS[1::-1]
    assert BUILTINS.__eq__(list(BUILTINS)) is NotImplemented
    assert repr(BUILTINS[:2]) == "<Methods (2)>"


def test_duplicate_references_are_rejected() -> None:
    with pytest.raises(MethodFileError, match=f"Duplicate method {BUILTINS[0].ref}."):
        Methods([*BUILTINS, BUILTINS[0]])
    files = [("sample/same", definition("gradeflow/full-marks@1"))] * 2
    with pytest.raises(MethodFileError, match=re.escape("Duplicate method sample/same@1.")):
        BUILTINS.with_files(files)


def test_missing_dependencies_and_cycles_are_reported() -> None:
    with pytest.raises(MethodFileError, match="sample/outer@1: There's no method missing@1"):
        BUILTINS.with_files([("sample/outer", definition("missing@1"))])
    with pytest.raises(MethodFileError, match="sample/outer@1: There's no method sample/inner"):
        BUILTINS.with_files(
            [
                ("sample/outer", definition("sample/inner")),
                ("sample/inner", definition("gradeflow/full-marks@1")),
            ]
        )
    with pytest.raises(MethodFileError, match="can't use itself"):
        BUILTINS.with_files(
            [
                ("sample/outer", definition("sample/inner@1")),
                ("sample/inner", definition("sample/outer@1")),
            ]
        )
    with pytest.raises(MethodFileError, match="no method missing@1"):
        BUILTINS.dependencies("missing@1")


class NoMarks(FullMarks):
    def _mark_answer(self, settings: Settings, answer: Answer, question: Question) -> Outcome:
        return Outcome(0, "Replacement.")


def result(methods: Methods, marking: object) -> float:
    questions = QuestionSet({"Q1": TextQuestion()})
    scheme = MarkScheme.load([marking], methods=methods)
    [submission] = mark(questions, scheme, [Submission(student_id="s", answers={"Q1": "x"})])
    return submission.results["Q1"].proportion


def test_composed_checks_use_the_current_collection_through_every_dependency() -> None:
    loaded = BUILTINS.with_files(
        [
            ("sample/outer", definition("sample/inner@1")),
            ("sample/inner", definition("gradeflow/full-marks@1")),
        ]
    )
    supplied = Methods(
        NoMarks() if method.name == "gradeflow/full-marks" else method for method in loaded
    )
    data = {"question": "Q1", "uses": "sample/outer@1"}
    assert result(loaded, data) == 1
    assert result(supplied, data) == 0
    # Existing checks are also rebound when included in a new operation.
    check = Check.load(data, methods=loaded)
    assert result(supplied, check) == 0
    incomplete = Methods(method for method in loaded if method.name != "gradeflow/full-marks")
    assert MarkScheme.load([data], methods=incomplete).problems(QuestionSet({"Q1": TextQuestion()}))
    with pytest.raises(MethodFileError, match="no method gradeflow/full-marks@1"):
        incomplete.dependencies("sample/outer@1")
    with pytest.raises(InvalidError, match="no method gradeflow/full-marks@1"):
        Check.load({"uses": "gradeflow/full-marks@1"}, methods=Methods())


def test_reference_rewriting_only_changes_named_references_and_copies_content() -> None:
    content = {"uses": "a", "with": {"text": "a", "checks": [{"uses": "b"}]}}
    changed = rewrite_references(content, {"a": "opaque@@", "b": "c"})
    assert methods_used(changed) == {"opaque@@", "c"}
    assert changed["with"]["text"] == "a"
    assert methods_used(content) == {"a", "b"}


def test_dependencies_include_default_checks_and_their_transitive_dependencies() -> None:
    wrapper = """label: Default checks
version: 1
summary: Uses the supplied checks.
question_types: [text]
inputs:
  rules:
    default:
      - uses: sample/leaf@1
checks: "{{ inputs.rules }}"
"""
    methods = BUILTINS.with_files(
        [("sample/wrapper", wrapper), ("sample/leaf", definition("gradeflow/full-marks@1"))]
    )
    assert methods.dependencies("sample/wrapper@1") == {
        "sample/leaf@1",
        "gradeflow/combination@1",
        "gradeflow/full-marks@1",
    }
    assert result(methods, {"question": "Q1", "uses": "sample/wrapper@1"}) == 1


def test_a_finite_nested_example_does_not_create_a_composition_cycle() -> None:
    from gradeflow_core import evaluate_examples

    wrapper = """label: Wrapper
version: 1
summary: A check with a default.
question_types: [text]
inputs:
  rule:
    default: {uses: gradeflow/full-marks@1}
checks:
  - "{{ inputs.rule }}"
examples:
  - title: A nested check
    description: A wrapper inside a wrapper.
    questions: {Q1: {type: text}}
    with:
      rule:
        uses: sample/wrapper@1
        with:
          rule: {uses: gradeflow/full-marks@1}
    rows:
      - answers: {Q1: sample}
        expected:
          Q1: {marks: 1, feedback: Full marks.}
"""
    methods = BUILTINS.with_files([("sample/wrapper", wrapper)])
    assert methods.dependencies("sample/wrapper@1") == {
        "gradeflow/combination@1",
        "gradeflow/full-marks@1",
    }
    [row] = evaluate_examples("sample/wrapper@1", methods=methods)[0].rows
    assert (row.total_marks, row.matches) == (1, True)
