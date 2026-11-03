from collections.abc import Iterator
from typing import Any

import pytest

from gradeflow_core import GradeFlowError, MarkScheme, QuestionSet, Submission, choose_models, mark
from gradeflow_core.editing import list_models, try_model
from gradeflow_core.questions.text import TextQuestion
from tests.fake_models import ENDPOINT, InstallFakeModels, Verdict, example_verdicts

QUESTIONS = QuestionSet({"Q1": TextQuestion(max_marks=3)})
SCHEME = MarkScheme.load(
    [
        {
            "question": "Q1",
            "uses": "gradeflow/answer-criteria@1",
            "with": {
                "criteria": [
                    {"criterion": "Correct total", "marks": 2},
                    {"criterion": "Shows the working", "marks": 1},
                    {"criterion": "Forgot to sum", "marks": -1},
                ]
            },
        }
    ]
)
CORRECT_ANSWER = "3 × 2 = 6 and 2 × 5 = 10, so 6 + 10 = $16"
UNSUMMED_ANSWER = "3 × 2 = 6 and 2 × 5 = 10, so she pays $6 and $10"
FAST_MODEL_ENDPOINT = ENDPOINT.model_copy(update={"fast_model": "decider"})


def submissions(*answers: str) -> list[Submission]:
    return [
        Submission(student_id=str(index), answers={"Q1": answer})
        for index, answer in enumerate(answers)
    ]


def test_results_stream_a_chunk_at_a_time(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    results: Iterator[Any] = mark(
        QUESTIONS,
        SCHEME,
        submissions(CORRECT_ANSWER, UNSUMMED_ANSWER, "$16"),
        model_endpoint=FAST_MODEL_ENDPOINT,
        chunk_size=2,
    )
    next(results)
    assert len(fake.requests) == 1
    list(results)
    assert len(fake.requests) == 2


def test_an_answer_asked_about_in_an_earlier_chunk_isnt_asked_again(
    fake_models: InstallFakeModels,
) -> None:
    fake = fake_models(example_verdicts)
    results = mark(
        QUESTIONS,
        SCHEME,
        submissions(CORRECT_ANSWER, UNSUMMED_ANSWER, CORRECT_ANSWER, UNSUMMED_ANSWER),
        model_endpoint=FAST_MODEL_ENDPOINT,
        chunk_size=2,
    )
    assert [submission.results["Q1"].proportion for submission in results] == [1, 0, 1, 0]
    assert len(fake.requests) == 1


def test_each_question_keeps_its_own_decisions(fake_models: InstallFakeModels) -> None:
    fake = fake_models(lambda model, answer, criterion: "no")
    fake.verdicts_by_question = {"How much does she pay?": lambda model, answer, criterion: "yes"}
    questions = QuestionSet(
        {
            "Q1": TextQuestion(max_marks=1, description="How much does she pay?"),
            "Q2": TextQuestion(max_marks=1, description="How much change does she get?"),
        }
    )
    scheme = MarkScheme.load(
        [
            {
                "question": question,
                "uses": "gradeflow/answer-criteria@1",
                "with": {"criteria": [{"criterion": "Correct total", "marks": 1}]},
            }
            for question in ("Q1", "Q2")
        ]
    )
    submission = Submission(student_id="1", answers={"Q1": "$16", "Q2": "$16"})
    [marked_submission] = mark(questions, scheme, [submission], model_endpoint=FAST_MODEL_ENDPOINT)
    results = marked_submission.results
    assert (results["Q1"].proportion, results["Q2"].proportion) == (1, 0)


def test_a_mark_scheme_says_whether_it_needs_a_model_endpoint() -> None:
    assert SCHEME.needs_model_endpoint()
    scheme_without_models = MarkScheme.load(
        [{"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {"answers": ["16"]}}]
    )
    assert not scheme_without_models.needs_model_endpoint()
    assert SCHEME.markings[0].needs_model_endpoint()
    assert not scheme_without_models.markings[0].needs_model_endpoint()


def test_choosing_models_keeps_those_that_mark_the_examples_as_expected(
    fake_models: InstallFakeModels,
) -> None:
    def verdicts(model: str, answer: str, criterion: str) -> Verdict:
        return "yes" if model == "careless" else example_verdicts(model, answer, criterion)

    fake = fake_models(verdicts)
    fake.models = ["careless", "decider", "reasoner"]
    fake.without_logprobs = {"reasoner"}
    choice = choose_models(ENDPOINT)
    assert choice.endpoint.fast_model == "decider"
    assert choice.endpoint.careful_model in {"decider", "reasoner"}
    trials = [
        (trial.model, trial.role, trial.rows_matched, trial.rows_tried, trial.failure)
        for trial in choice.trials
    ]
    assert trials == [
        ("careless", "fast", 0, 9, "Missed 9 of 9 example answers."),
        ("decider", "fast", 9, 9, None),
        ("reasoner", "fast", 0, 0, "Doesn't return token probabilities."),
        ("careless", "careful", 0, 9, "Missed 9 of 9 example answers."),
        ("decider", "careful", 9, 9, None),
        ("reasoner", "careful", 9, 9, None),
    ]
    assert all(trial.seconds >= 0 for trial in choice.trials)


def test_choosing_models_says_why_a_model_fails(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    fake.models = ["decider"]
    fake.listed_models = ["gone", "decider"]
    choice = choose_models(ENDPOINT)
    assert [trial.failure for trial in choice.trials if trial.model == "gone"] == [
        "The AI service has no such model."
    ] * 2


def test_choosing_models_at_an_address_with_none_says_so(fake_models: InstallFakeModels) -> None:
    fake_models(example_verdicts)
    with pytest.raises(GradeFlowError, match="No models were found at this address"):
        choose_models(ENDPOINT.model_copy(update={"url": "https://models.test/wrong"}))


def test_listing_models_gives_the_endpoints_models(fake_models: InstallFakeModels) -> None:
    fake_models(example_verdicts).listed_models = ["decider", "reasoner"]
    assert list_models(ENDPOINT) == ["decider", "reasoner"]


def test_trying_one_model_says_how_it_did(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    fake.without_logprobs = {"reasoner"}
    trial = try_model(ENDPOINT, "fast", "decider")
    assert (trial.rows_matched, trial.rows_tried, trial.failure) == (9, 9, None)
    assert try_model(ENDPOINT, "fast", "reasoner").failure == "Doesn't return token probabilities."
