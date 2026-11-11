import re

import pytest

from gradeflow_core import (
    BUILTINS,
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
from gradeflow_core.editing import dump_yaml, methods_used, rewrite_references
from gradeflow_core.methods.logic.full_marks import FullMarks
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.types import Answer
from gradeflow_core.steps.method import AnswerMethod
from gradeflow_core.steps.outcome import Outcome
from gradeflow_core.steps.settings import Settings
from gradeflow_core.steps.step import valid_step


def method_file(uses: str, version: int = 1) -> str:
    return dump_yaml(
        {
            "label": "Wrapper",
            "version": version,
            "summary": "A wrapper.",
            "question_types": ["text"],
            "steps": [{"uses": uses}],
        }
    )


def test_methods_find_a_method_by_its_ref() -> None:
    assert BUILTINS.find("gradeflow/key-points@1") is BUILTINS.by_ref["gradeflow/key-points@1"]
    assert repr(BUILTINS.find("gradeflow/key-points@1")) == "<Method gradeflow/key-points@1>"
    with pytest.raises(GradeFlowError, match="There's no method gradeflow/key-points@9"):
        BUILTINS.find("gradeflow/key-points@9")
    assert len(Methods()) == 0


def test_a_check_uses_the_built_in_methods_unless_given_others() -> None:
    full_marks = BUILTINS.find("gradeflow/full-marks@1")
    assert valid_step({"uses": full_marks.ref}, BUILTINS.by_ref).resolution.method is full_marks


def test_loading_preserves_setting_order_and_does_not_modify_supplied_collections() -> None:
    original = list(BUILTINS)
    files = [
        ("sample/outer", method_file("sample/inner@1")),
        ("sample/inner", method_file("gradeflow/full-marks@1")),
    ]
    methods = BUILTINS.with_files(files)
    assert list(BUILTINS) == original
    assert list(methods[:-2]) == original
    assert [method.ref for method in methods[-2:]] == ["sample/outer@1", "sample/inner@1"]
    assert methods.dependencies("sample/outer@1") == {
        "sample/inner@1",
        "gradeflow/workflow@1",
        "gradeflow/full-marks@1",
    }


def test_versions_of_a_method_are_distinct_and_latest_keeps_the_highest() -> None:
    files = [
        ("sample/wrapper", method_file("gradeflow/full-marks@1", version=2)),
        ("sample/other", method_file("gradeflow/full-marks@1")),
        ("sample/wrapper", method_file("gradeflow/not-allowed@1", version=1)),
    ]
    methods = BUILTINS.with_files(files)
    loaded = Methods(methods[len(BUILTINS) :])
    assert [method.ref for method in loaded] == [
        "sample/wrapper@2",
        "sample/other@1",
        "sample/wrapper@1",
    ]
    assert {method.name for method in loaded} == {"sample/wrapper", "sample/other"}
    assert valid_step({"uses": "sample/wrapper@1"}, methods.by_ref).resolution.method is loaded[2]
    assert valid_step({"uses": "sample/wrapper@2"}, methods.by_ref).resolution.method is loaded[0]
    with pytest.raises(InvalidError, match="There's no method sample/wrapper"):
        valid_step({"uses": "sample/wrapper"}, methods.by_ref)
    assert [method.ref for method in loaded.latest()] == ["sample/wrapper@2", "sample/other@1"]
    assert list(BUILTINS.latest()) == list(BUILTINS)


def test_duplicate_references_are_rejected() -> None:
    with pytest.raises(InvalidError, match=f"Duplicate method {BUILTINS[0].ref}."):
        Methods([*BUILTINS, BUILTINS[0]])
    files = [("sample/same", method_file("gradeflow/full-marks@1"))] * 2
    with pytest.raises(MethodFileError, match=re.escape("sample/same@1: Duplicate method.")):
        BUILTINS.with_files(files)


def test_a_method_whose_settings_share_a_steps_property_is_refused() -> None:
    class NamedSettings(Settings):
        name: str = ""
        weight: float = 1

    class Named(AnswerMethod[NamedSettings]):
        name, version, label, summary = "sample/named", 1, "Named", "Named."
        settings = NamedSettings

        def describe(self, settings: NamedSettings) -> str:
            return ""

        def mark_answer(
            self, settings: NamedSettings, answer: Answer, question: Question
        ) -> Outcome:
            return Outcome(1, "")

    with pytest.raises(
        TypeError, match=re.escape("Named's settings share a step's properties: ['name', 'weight']")
    ):
        Methods([*BUILTINS, Named()])


def test_missing_dependencies_and_cycles_are_reported() -> None:
    with pytest.raises(
        MethodFileError, match="sample/outer@1: Line 7: There's no method missing@1"
    ):
        BUILTINS.with_files([("sample/outer", method_file("missing@1"))])
    with pytest.raises(
        MethodFileError, match="sample/outer@1: Line 7: There's no method sample/inner"
    ):
        BUILTINS.with_files(
            [
                ("sample/outer", method_file("sample/inner")),
                ("sample/inner", method_file("gradeflow/full-marks@1")),
            ]
        )
    with pytest.raises(MethodFileError, match="sample/outer@1: Can't use itself"):
        BUILTINS.with_files(
            [
                ("sample/outer", method_file("sample/inner@1")),
                ("sample/inner", method_file("sample/outer@1")),
            ]
        )
    with pytest.raises(InvalidError, match="no method missing@1"):
        BUILTINS.dependencies("missing@1")


class ZeroFullMarks(FullMarks):
    def mark_answer(self, settings: Settings, answer: Answer, question: Question) -> Outcome:
        return Outcome(0, "Replacement.")


def proportion_with(methods: Methods, marking: object) -> float:
    questions = QuestionSet({"Q1": TextQuestion()})
    scheme = MarkScheme.load([marking], methods=methods)
    [submission] = mark(questions, scheme, [Submission(student_id="s", answers={"Q1": "x"})])
    return submission.results["Q1"].proportion


def test_composed_checks_use_the_current_collection_through_every_dependency() -> None:
    loaded = BUILTINS.with_files(
        [
            ("sample/outer", method_file("sample/inner@1")),
            ("sample/inner", method_file("gradeflow/full-marks@1")),
        ]
    )
    with_replaced_full_marks = Methods(
        ZeroFullMarks() if method.name == "gradeflow/full-marks" else method for method in loaded
    )
    marking = {"question": "Q1", "uses": "sample/outer@1"}
    assert proportion_with(loaded, marking) == 1
    assert proportion_with(with_replaced_full_marks, marking) == 0
    # Existing checks are also rebound when included in a new operation.
    check = valid_step(marking, loaded.by_ref)
    assert proportion_with(with_replaced_full_marks, check) == 0
    without_full_marks = Methods(
        method for method in loaded if method.name != "gradeflow/full-marks"
    )
    assert MarkScheme.load([marking], methods=without_full_marks).problems(
        QuestionSet({"Q1": TextQuestion()})
    )
    with pytest.raises(InvalidError, match="no method gradeflow/full-marks@1"):
        without_full_marks.dependencies("sample/outer@1")
    with pytest.raises(InvalidError, match="no method gradeflow/full-marks@1"):
        valid_step({"uses": "gradeflow/full-marks@1"}, Methods().by_ref)


def test_reference_rewriting_only_changes_named_references_and_copies_content() -> None:
    check = {"uses": "a", "with": {"text": "a", "checks": [{"uses": "b"}]}}
    rewritten = rewrite_references(check, {"a": "opaque@@", "b": "c"})
    assert methods_used(rewritten) == {"opaque@@", "c"}
    assert rewritten["with"]["text"] == "a"
    assert methods_used(check) == {"a", "b"}


def test_dependencies_include_default_checks_and_their_transitive_dependencies() -> None:
    wrapper_file = """label: Default checks
version: 1
summary: Uses the supplied checks.
question_types: [text]
settings:
  rules:
    default:
      - uses: sample/inner@1
steps: "{{ settings.rules }}"
"""
    methods = BUILTINS.with_files(
        [("sample/wrapper", wrapper_file), ("sample/inner", method_file("gradeflow/full-marks@1"))]
    )
    assert methods.dependencies("sample/wrapper@1") == {
        "sample/inner@1",
        "gradeflow/workflow@1",
        "gradeflow/full-marks@1",
    }
    assert proportion_with(methods, {"question": "Q1", "uses": "sample/wrapper@1"}) == 1


def test_a_method_needs_a_model_endpoint_when_a_default_check_does() -> None:
    wrapper_file = """label: Default checks
version: 1
summary: Uses the supplied checks.
question_types: [text]
settings:
  rules:
    default:
      - uses: gradeflow/answer-criteria@1
        with: {criteria: [{text: Names a cell.}]}
steps: "{{ settings.rules }}"
"""
    methods = BUILTINS.with_files([("sample/wrapper", wrapper_file)])
    assert methods.by_ref["sample/wrapper@1"].needs_model_endpoint


def test_a_finite_nested_example_does_not_create_a_composition_cycle() -> None:
    from gradeflow_core import mark_examples

    wrapper_file = """label: Wrapper
version: 1
summary: A check with a default.
question_types: [text]
settings:
  rule:
    default: {uses: gradeflow/full-marks@1}
steps:
  - "{{ settings.rule }}"
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
    methods = BUILTINS.with_files([("sample/wrapper", wrapper_file)])
    assert methods.dependencies("sample/wrapper@1") == {
        "gradeflow/workflow@1",
        "gradeflow/full-marks@1",
    }
    [row] = mark_examples("sample/wrapper@1", methods=methods)[0].rows
    assert (row.total_marks, row.matches_expected) == (1, True)
