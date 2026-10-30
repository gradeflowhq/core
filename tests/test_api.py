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
        "ChoiceQuestion",
        "CodeQuestion",
        "GradeFlowError",
        "InvalidError",
        "MarkScheme",
        "MarkedExample",
        "MarkedSheet",
        "MarkedSubmission",
        "Method",
        "MethodFileError",
        "Methods",
        "ModelChoice",
        "ModelEndpoint",
        "NumberQuestion",
        "Problem",
        "Progress",
        "Question",
        "QuestionResult",
        "QuestionSet",
        "Report",
        "ResultsCsv",
        "Submission",
        "TextQuestion",
        "ValuesQuestion",
        "choose_models",
        "compare_answers",
        "evaluate_examples",
        "mark",
        "read_answer_key",
        "read_question_document",
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
    public = getattr(module, name)
    if isinstance(public, str):
        return
    pydoc.render_doc(public)
    if inspect.isfunction(public) or (
        inspect.isclass(public) and not issubclass(public, BaseException)
    ):
        inspect.signature(public)


def test_import_leaves_the_language_model_unloaded() -> None:
    code = (
        "import sys, gradeflow_core, gradeflow_core.editing\n"
        "assert not {'fastembed', 'numpy', 'onnxruntime', 'openai'} & sys.modules.keys()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_imported_answers_are_marked_and_unmarked_questions_keep_their_given_marks() -> None:
    students = AnswersCsv(
        student_id_column="student_id", answer_columns=["Q1", "Q2"], marks_columns={"Q2": "m2"}
    ).read("student_id,Q1,Q2,m2\ns1,hello,x,0.5\ns2,world,y,1\n")
    marked = {
        submission.student_id: submission.results
        for submission in mark(QUESTIONS, SCHEME, students)
    }
    assert {
        key: {q: r.proportion for q, r in results.items()} for key, results in marked.items()
    } == {
        "s1": {"Q1": 1, "Q2": 0.5},
        "s2": {"Q1": 0, "Q2": 1},
    }
    assert [student.marks for student in students] == [{"Q2": 0.5}, {"Q2": 1}]


def test_mark_yields_each_student_once_they_are_marked() -> None:
    submissions = (
        Submission(student_id=f"s{index}", answers={"Q1": "hello"}) for index in range(3)
    )
    marked = mark(QUESTIONS, SCHEME, submissions)
    assert next(marked).student_id == "s0"


def test_a_sheet_is_marked_as_a_sheet_and_a_submission_as_a_submission() -> None:
    sheets = [
        AnswerSheet(answers={"Q1": "hello"}, marks={"Q2": 0.5}),
        Submission(student_id="s1", answers={"Q1": "world"}),
    ]
    sheet, student = mark(QUESTIONS, SCHEME, sheets)
    assert type(sheet) is MarkedSheet
    assert {key: result.proportion for key, result in sheet.results.items()} == {"Q1": 1, "Q2": 0.5}
    assert isinstance(student, MarkedSubmission)
    assert student.student_id == "s1"


def test_what_grading_takes_round_trips() -> None:
    submission = Submission(student_id="s1", answers={"Q1": "x"})
    assert QuestionSet.model_validate_json(QUESTIONS.model_dump_json()) == QUESTIONS
    assert MarkScheme.from_yaml(SCHEME.to_yaml()) == SCHEME
    assert Submission.model_validate_json(submission.model_dump_json()) == submission
