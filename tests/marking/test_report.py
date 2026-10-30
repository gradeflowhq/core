from gradeflow_core import MarkedSubmission, QuestionResult, Report

MARKED = MarkedSubmission(
    student_id="s1",
    answers={},
    results={
        "Q10": QuestionResult(proportion=1, max_marks=1, feedback="Correct."),
        "Q2": QuestionResult(proportion=0.5, max_marks=3, feedback="  "),
    },
)


def test_report_gives_each_question_its_marks_then_feedback_in_natural_order() -> None:
    found = MARKED.report()
    assert found == [Report("Q2", 1.5, 3, ""), Report("Q10", 1, 1, "Correct.")]
    assert [entry.heading for entry in found] == ["Q2 [1.5/3]", "Q10 [1/1]"]
    assert "\n".join(map(str, found)) == "Q2 [1.5/3]\nQ10 [1/1]\nCorrect."
