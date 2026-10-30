import subprocess
import sys
import textwrap


def test_grading_and_editing_without_the_llm_extra() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import sys
            sys.modules["openai"] = None

            from gradeflow_core import (
                GradeFlowError, MarkScheme, ModelEndpoint, QuestionSet, Submission,
                choose_models, mark,
            )
            from gradeflow_core.editing import MarkingDraft, list_models, try_model

            questions = QuestionSet.load({"Q1": {"type": "text"}})
            students = [Submission(student_id="1", answers={"Q1": "hello"})]
            scheme = MarkScheme.load([
                {"question": "Q1", "uses": "gradeflow/accepted-answers@1",
                 "with": {"answers": ["hello"]}},
            ])
            [marked] = mark(questions, scheme, students)
            assert marked.results["Q1"].correct
            draft = MarkingDraft(scheme.dump()[0], questions, "Q1")
            assert draft.problems == []

            endpoint = ModelEndpoint(
                url="https://models.test/v1", token="test",
                decision_model="decider", reasoning_model="reasoner",
            )
            for operation in (list_models, choose_models):
                try:
                    operation(endpoint)
                except GradeFlowError as error:
                    assert "gradeflow-core[llm]" in str(error), str(error)
                else:
                    raise AssertionError("Missing LLM extra was not reported")

            for role in ("decision", "reasoning"):
                trial = try_model(endpoint, role, "model")
                assert "gradeflow-core[llm]" in trial.failure, trial

            for mode in ("decision", "reasoning", "both"):
                scheme = MarkScheme.load([
                    {"question": "Q1", "uses": "gradeflow/answer-criteria@1",
                     "with": {"criteria": [{"text": "Says hello", "marks": 1}],
                              "decide_with": mode}},
                ])
                [marked] = mark(questions, scheme, students, model_endpoint=endpoint)
                result = marked.results["Q1"]
                assert not result.marked, result
                assert "gradeflow-core[llm]" in result.feedback, result.feedback

                [marked] = mark(questions, scheme, students)
                assert not marked.results["Q1"].marked
                assert "model endpoint" in marked.results["Q1"].feedback
            """),
        ],
        check=True,
    )
