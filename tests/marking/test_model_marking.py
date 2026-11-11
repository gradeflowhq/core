from collections.abc import Iterator
from typing import Any

import pytest

from gradeflow_core import (
    GradeFlowError,
    MarkScheme,
    ModelTrial,
    QuestionSet,
    Submission,
    best_models,
    list_models,
    mark,
    try_model,
    try_models,
)
from gradeflow_core.methods.text.answer_criteria import ModelRole
from gradeflow_core.questions.text import TextQuestion
from tests.fake_models import (
    ENDPOINT,
    FAST_MODEL_ENDPOINT,
    InstallFakeModels,
    Verdict,
    example_verdicts,
)

QUESTIONS = QuestionSet({"Q1": TextQuestion(max_marks=3)})
SCHEME = MarkScheme.load(
    [
        {
            "question": "Q1",
            "uses": "gradeflow/answer-criteria@1",
            "with": {
                "criteria": [
                    {"text": "Correct total", "marks": 2},
                    {"text": "Shows the working", "marks": 1},
                    {"text": "Forgot to sum", "marks": -1},
                ]
            },
        }
    ]
)
CORRECT_ANSWER = "3 × 2 = 6 and 2 × 5 = 10, so 6 + 10 = $16"
UNSUMMED_ANSWER = "3 × 2 = 6 and 2 × 5 = 10, so she pays $6 and $10"


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
                "question": question_id,
                "uses": "gradeflow/answer-criteria@1",
                "with": {"criteria": [{"text": "Correct total", "marks": 1}]},
            }
            for question_id in ("Q1", "Q2")
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


def test_trying_models_tries_each_as_the_fast_model_then_the_careful_one(
    fake_models: InstallFakeModels,
) -> None:
    def verdicts(model: str, answer: str, criterion: str) -> Verdict:
        return "yes" if model == "careless" else example_verdicts(model, answer, criterion)

    fake = fake_models(verdicts)
    fake.models = ["careless", "decider", "reasoner"]
    fake.without_logprobs = {"reasoner"}
    trials = list(try_models(ENDPOINT))
    assert [
        (trial.model, trial.role, trial.rows_matched, trial.rows_tried, trial.failure)
        for trial in trials
    ] == [
        ("careless", "fast", 1, 12, "Missed 11 of 12 example answers."),
        ("decider", "fast", 12, 12, None),
        ("reasoner", "fast", 0, 0, "Doesn't return token probabilities."),
        ("careless", "careful", 1, 12, "Missed 11 of 12 example answers."),
        ("decider", "careful", 12, 12, None),
        ("reasoner", "careful", 12, 12, None),
    ]
    assert all(trial.seconds >= 0 for trial in trials)


def test_trying_models_stops_once_two_pass_in_each_role(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    fake.models = ["first", "second", "third"]
    fake.without_logprobs = {"first"}
    tried = [(trial.model, trial.role) for trial in try_models(ENDPOINT)]
    assert tried == [
        ("first", "fast"),
        ("second", "fast"),
        ("third", "fast"),
        ("first", "careful"),
        ("second", "careful"),
    ]


def test_trying_models_skips_those_tried_before_and_counts_their_passes(
    fake_models: InstallFakeModels,
) -> None:
    fake = fake_models(example_verdicts)
    fake.models = ["first", "second", "third"]
    fake.without_logprobs = {"first"}
    passes: list[tuple[str, ModelRole]] = [("second", "fast"), ("first", "careful")]
    tried = [
        ModelTrial(model=name, role=role, rows_matched=9, rows_tried=9, seconds=1, failure=None)
        for name, role in passes
    ]
    assert [(trial.model, trial.role) for trial in try_models(ENDPOINT, tried)] == [
        ("first", "fast"),
        ("third", "fast"),
        ("second", "careful"),
    ]


def test_trying_models_tries_again_those_that_errored(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    fake.models = ["decider"]
    errored = ModelTrial(
        model="decider",
        role="fast",
        rows_matched=0,
        rows_tried=None,
        seconds=0.1,
        failure="The AI service answered 503.",
    )
    assert [(trial.model, trial.role) for trial in try_models(ENDPOINT, [errored])] == [
        ("decider", "fast"),
        ("decider", "careful"),
    ]


def test_trying_models_skips_those_that_refused_a_role(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    fake.models = ["reasoner"]
    fake.without_logprobs = {"reasoner"}
    [refused, passed] = try_models(ENDPOINT)
    assert (refused.role, refused.rows_tried, passed.role) == ("fast", 0, "careful")
    assert list(try_models(ENDPOINT, [refused, passed])) == []
    assert best_models([refused, passed]) == (None, "reasoner")


def test_trying_models_says_when_each_test_starts(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    fake.models = ["decider"]
    started: list[tuple[str, ModelRole]] = []
    trials = try_models(ENDPOINT, on_start=lambda model, role: started.append((model, role)))
    assert next(trials).role == "fast"
    assert started == [("decider", "fast")]
    assert [trial.role for trial in trials] == ["careful"]
    assert started == [("decider", "fast"), ("decider", "careful")]


def test_trying_models_says_why_a_model_fails(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    fake.models = ["decider"]
    fake.listed_models = ["gone", "decider"]
    assert [
        (trial.rows_tried, trial.failure) for trial in try_models(ENDPOINT) if trial.model == "gone"
    ] == [(None, "The AI service has no such model.")] * 2


def test_trying_models_at_an_address_with_none_says_so(fake_models: InstallFakeModels) -> None:
    fake_models(example_verdicts)
    with pytest.raises(GradeFlowError, match="No models were found at this address"):
        next(try_models(ENDPOINT.model_copy(update={"url": "https://models.test/wrong"})))


def test_the_best_models_have_the_most_right_then_are_fastest() -> None:
    def trial(model: str, role: ModelRole, seconds: float, rows_matched: int = 9) -> ModelTrial:
        rows_tried = 0 if rows_matched < 0 else 9
        return ModelTrial(
            model=model,
            role=role,
            rows_matched=max(rows_matched, 0),
            rows_tried=rows_tried,
            seconds=seconds,
            failure=None if rows_matched == rows_tried else "Missed some.",
        )

    trials = [
        trial("slow", "fast", 9),
        trial("quick", "fast", 3),
        trial("quicker", "fast", 1, rows_matched=8),
        trial("slow", "careful", 30),
    ]
    assert best_models(trials) == ("quick", "slow")
    assert best_models(trials[2:3]) == ("quicker", None), "the best so far needn't get them all"
    partial = [trial("six", "fast", 5, rows_matched=6), trial("seven", "fast", 9, rows_matched=7)]
    assert best_models(partial) == ("seven", None)
    assert best_models([trial("broken", "fast", 0, rows_matched=-1)]) == (None, None), (
        "one that never ran"
    )
    assert best_models([]) == (None, None)


def test_listing_models_gives_the_endpoints_models(fake_models: InstallFakeModels) -> None:
    fake_models(example_verdicts).listed_models = ["decider", "reasoner"]
    assert list_models(ENDPOINT) == ["decider", "reasoner"]


def test_trying_one_model_says_how_it_did(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    fake.without_logprobs = {"reasoner"}
    trial = try_model(ENDPOINT, "fast", "decider")
    assert (trial.rows_matched, trial.rows_tried, trial.failure) == (12, 12, None)
    assert try_model(ENDPOINT, "fast", "reasoner").failure == "Doesn't return token probabilities."
