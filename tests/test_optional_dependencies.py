import subprocess
import sys
import textwrap


def test_marking_and_editing_without_the_llm_extra() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import sys
            sys.modules["openai"] = None

            from gradeflow_core import (
                GradeFlowError, MarkScheme, ModelEndpoint, QuestionSet, Submission,
                list_models, mark, try_model, try_models,
            )
            from gradeflow_core.editing import MarkingDraft

            questions = QuestionSet.load({"Q1": {"type": "text"}})
            submissions = [Submission(student_id="1", answers={"Q1": "hello"})]
            scheme = MarkScheme.load([
                {"question": "Q1", "uses": "gradeflow/accepted-answers@1",
                 "with": {"answers": ["hello"]}},
            ])
            [marked] = mark(questions, scheme, submissions)
            assert marked.results["Q1"].correct
            draft = MarkingDraft(scheme.dump()[0], questions, "Q1")
            assert draft.problems == []

            endpoint = ModelEndpoint(
                url="https://models.test/v1", api_key="test",
                fast_model="decider", careful_model="reasoner",
            )
            for operation in (list_models, lambda endpoint: next(try_models(endpoint))):
                try:
                    operation(endpoint)
                except GradeFlowError as error:
                    assert "gradeflow-core[llm]" in str(error), str(error)
                else:
                    raise AssertionError("Missing LLM extra was not reported")

            for role in ("fast", "careful"):
                trial = try_model(endpoint, role, "model")
                assert "gradeflow-core[llm]" in trial.failure, trial

            for mode in ("fast", "careful", "fast_then_careful"):
                scheme = MarkScheme.load([
                    {"question": "Q1", "uses": "gradeflow/answer-criteria@1",
                     "with": {"criteria": [{"text": "Says hello", "marks": 1}],
                              "decide_with": mode}},
                ])
                [marked] = mark(questions, scheme, submissions, model_endpoint=endpoint)
                result = marked.results["Q1"]
                assert not result.is_marked, result
                assert "gradeflow-core[llm]" in result.feedback, result.feedback

                [marked] = mark(questions, scheme, submissions)
                assert not marked.results["Q1"].is_marked
                assert "no AI service" in marked.results["Q1"].feedback
            """),
        ],
        check=True,
    )
