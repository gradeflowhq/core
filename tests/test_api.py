import inspect
import pydoc
import subprocess
import sys
from types import ModuleType

import pytest

import gradeflow_core
from gradeflow_core import (
    AnswersCsv,
    AnswerSheet,
    Marked,
    MarkScheme,
    QuestionSet,
    Submission,
    TextQuestion,
    editing,
    mark,
)

QUESTIONS = QuestionSet({"Q1": TextQuestion(), "Q2": TextQuestion()})
SCHEME = MarkScheme.load(
    [{"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {"answers": ["hello"]}}]
)


def test_scripting_needs_only_these_names() -> None:
    assert sorted(gradeflow_core.__all__) == [
        "Answer",
        "AnswerComparison",
        "AnswerMethod",
        "AnswerSheet",
        "AnswersCsv",
        "BUILTINS",
        "CheckResult",
        "ChoiceQuestion",
        "CodeQuestion",
        "Detail",
        "GradeFlowError",
        "InvalidError",
        "MarkScheme",
        "Marked",
        "MarkedExample",
        "Method",
        "MethodFileError",
        "Methods",
        "ModelEndpoint",
        "ModelTrial",
        "NumberQuestion",
        "Outcome",
        "Problem",
        "Progress",
        "Question",
        "QuestionResult",
        "QuestionSet",
        "ReportLine",
        "ResultsCsv",
        "Settings",
        "Step",
        "Submission",
        "TextQuestion",
        "TransformMethod",
        "Unreadable",
        "ValuesQuestion",
        "answer_from_files",
        "best_models",
        "compare_answers",
        "list_models",
        "mark",
        "mark_examples",
        "read_answer_key",
        "read_question_document",
        "try_model",
        "try_models",
    ]
    assert not set(gradeflow_core.__all__) & set(editing.__all__)


@pytest.mark.parametrize("module", [gradeflow_core, editing], ids=lambda module: module.__name__)
def test_every_public_name_is_importable(module: ModuleType) -> None:
    namespace: dict[str, object] = {}
    exec(f"from {module.__name__} import *", namespace)
    assert set(module.__all__) <= namespace.keys()


@pytest.mark.parametrize(
    ("module", "name"),
    [(module, name) for module in (gradeflow_core, editing) for name in module.__all__],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_every_public_name_has_help_and_a_signature(module: ModuleType, name: str) -> None:
    member = getattr(module, name)
    if isinstance(member, str):
        return
    pydoc.render_doc(member)
    if inspect.isfunction(member) or (
        inspect.isclass(member) and not issubclass(member, BaseException)
    ):
        inspect.signature(member)


def test_import_leaves_the_language_models_unloaded() -> None:
    optional_packages = {
        "fastembed",
        "huggingface_hub",
        "numpy",
        "onnxruntime",
        "openai",
        "tokenizers",
    }
    code = (
        "import sys, gradeflow_core, gradeflow_core.editing\n"
        f"assert not {optional_packages} & sys.modules.keys()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_imported_answers_are_marked_and_unmarked_questions_keep_their_given_marks() -> None:
    submissions = AnswersCsv(
        student_id_column="student_id", answer_columns=["Q1", "Q2"], marks_columns={"Q2": "m2"}
    ).read("student_id,Q1,Q2,m2\ns1,hello,x,0.5\ns2,world,y,1\n")
    results_by_student = {
        submission.sheet.student_id: submission.results
        for submission in mark(QUESTIONS, SCHEME, submissions)
    }
    assert {
        student_id: {question_id: result.proportion for question_id, result in results.items()}
        for student_id, results in results_by_student.items()
    } == {
        "s1": {"Q1": 1, "Q2": 0.5},
        "s2": {"Q1": 0, "Q2": 1},
    }
    assert [submission.marks for submission in submissions] == [{"Q2": 0.5}, {"Q2": 1}]


def test_mark_yields_each_student_once_they_are_marked() -> None:
    submissions = (
        Submission(student_id=f"s{index}", answers={"Q1": "hello"}) for index in range(3)
    )
    marked_submissions = mark(QUESTIONS, SCHEME, submissions)
    assert next(marked_submissions).sheet.student_id == "s0"


def test_a_marked_sheet_holds_the_sheet_it_was_given() -> None:
    sheets = [
        AnswerSheet(answers={"Q1": "hello"}, marks={"Q2": 0.5}),
        Submission(student_id="s1", answers={"Q1": "world"}),
    ]
    marked_sheet, marked_submission = mark(QUESTIONS, SCHEME, sheets)
    assert [marked_sheet.sheet, marked_submission.sheet] == sheets
    assert {
        question_id: result.proportion for question_id, result in marked_sheet.results.items()
    } == {"Q1": 1, "Q2": 0.5}


def test_marked_submissions_round_trip() -> None:
    submissions = [Submission(student_id="s1", answers={"Q1": "hello"})]
    [marked] = mark(QUESTIONS, SCHEME, submissions)
    assert Marked[Submission].model_validate_json(marked.model_dump_json()) == marked


def test_marking_inputs_round_trip() -> None:
    submission = Submission(student_id="s1", answers={"Q1": "x"})
    assert QuestionSet.model_validate_json(QUESTIONS.model_dump_json()) == QUESTIONS
    assert MarkScheme.from_yaml(SCHEME.to_yaml()) == SCHEME
    assert Submission.model_validate_json(submission.model_dump_json()) == submission
