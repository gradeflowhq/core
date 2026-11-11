from gradeflow_core import Marked, QuestionResult, ReportLine, Submission

MARKED_SUBMISSION = Marked(
    sheet=Submission(student_id="s1", answers={}),
    results={
        "Q10": QuestionResult(proportion=1, max_marks=1, feedback="Correct."),
        "Q2": QuestionResult(proportion=0.5, max_marks=3, feedback="  "),
    },
    max_marks=4,
)


def test_report_gives_each_question_its_marks_then_feedback_in_the_results_order() -> None:
    reports = MARKED_SUBMISSION.report()
    assert reports == [ReportLine("Q10 [1/1]", "Correct."), ReportLine("Q2 [1.5/3]", "")]
    assert [report.heading for report in reports] == ["Q10 [1/1]", "Q2 [1.5/3]"]
    assert "\n".join(map(str, reports)) == "Q10 [1/1]\nCorrect.\nQ2 [1.5/3]"
