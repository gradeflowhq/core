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
    MarkedSheet,
    MarkedSubmission,
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
        "AnswerComparison",
        "AnswerSheet",
        "AnswersCsv",
        "BUILTINS",
        "Check",
        "CheckResult",
        "ChoiceQuestion",
        "CodeQuestion",
        "Detail",
        "GradeFlowError",
        "InvalidError",
        "MarkScheme",
        "MarkedExample",
        "MarkedSheet",
        "MarkedSubmission",
        "Method",
        "MethodFileError",
        "Methods",
        "ModelEndpoint",
        "ModelTrial",
        "NumberQuestion",
        "Problem",
        "Progress",
        "Question",
        "QuestionResult",
        "QuestionSet",
        "ReportLine",
        "ResultsCsv",
        "Submission",
        "TextQuestion",
        "ValuesQuestion",
        "answer_from_files",
        "best_models",
        "compare_answers",
        "mark",
        "mark_examples",
        "read_answer_key",
        "read_question_document",
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
        submission.student_id: submission.results
        for submission in mark(QUESTIONS, SCHEME, submissions)
    }
    assert {
        student_id: {question: result.proportion for question, result in results.items()}
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
    assert next(marked_submissions).student_id == "s0"


def test_a_sheet_is_marked_as_a_sheet_and_a_submission_as_a_submission() -> None:
    sheets = [
        AnswerSheet(answers={"Q1": "hello"}, marks={"Q2": 0.5}),
        Submission(student_id="s1", answers={"Q1": "world"}),
    ]
    sheet, submission = mark(QUESTIONS, SCHEME, sheets)
    assert type(sheet) is MarkedSheet
    assert {question_id: result.proportion for question_id, result in sheet.results.items()} == {
        "Q1": 1,
        "Q2": 0.5,
    }
    assert isinstance(submission, MarkedSubmission)
    assert submission.student_id == "s1"


def test_marking_inputs_round_trip() -> None:
    submission = Submission(student_id="s1", answers={"Q1": "x"})
    assert QuestionSet.model_validate_json(QUESTIONS.model_dump_json()) == QUESTIONS
    assert MarkScheme.from_yaml(SCHEME.to_yaml()) == SCHEME
    assert Submission.model_validate_json(submission.model_dump_json()) == submission
