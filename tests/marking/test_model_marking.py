from collections.abc import Callable, Iterator
from typing import Any

import pytest

from gradeflow_core import GradeFlowError, MarkScheme, QuestionSet, Submission, choose_models, mark
from gradeflow_core.editing import list_models, try_model
from gradeflow_core.questions.text import TextQuestion
from tests.fake_models import ENDPOINT, FakeModels, Oracle, Verdict, examples_oracle

type Install = Callable[[Oracle], FakeModels]

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
RIGHT = "3 × 2 = 6 and 2 × 5 = 10, so 6 + 10 = $16"
UNSUMMED = "3 × 2 = 6 and 2 × 5 = 10, so she pays $6 and $10"
DECIDING = ENDPOINT.model_copy(update={"decision_model": "decider"})


def submissions(*answers: str) -> list[Submission]:
    return [
        Submission(student_id=str(n), answers={"Q1": answer}) for n, answer in enumerate(answers)
    ]


def test_results_stream_a_chunk_at_a_time(fake_models: Install) -> None:
    fake = fake_models(examples_oracle)
    results: Iterator[Any] = mark(
        QUESTIONS,
        SCHEME,
        submissions(RIGHT, UNSUMMED, "$16"),
        model_endpoint=DECIDING,
        chunk_size=2,
    )
    next(results)
    assert len(fake.requests) == 1
    list(results)
    assert len(fake.requests) == 2


def test_an_answer_asked_about_in_an_earlier_chunk_isnt_asked_again(fake_models: Install) -> None:
    fake = fake_models(examples_oracle)
    results = mark(
        QUESTIONS,
        SCHEME,
        submissions(RIGHT, UNSUMMED, RIGHT, UNSUMMED),
        model_endpoint=DECIDING,
        chunk_size=2,
    )
    assert [result.results["Q1"].proportion for result in results] == [1, 0, 1, 0]
    assert len(fake.requests) == 1


def test_each_question_keeps_its_own_decisions(fake_models: Install) -> None:
    fake = fake_models(lambda model, answer, criterion: "no")
    fake.by_question = {"How much does she pay?": lambda model, answer, criterion: "yes"}
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
                "with": {"criteria": [{"text": "Correct total", "marks": 1}]},
            }
            for question in ("Q1", "Q2")
        ]
    )
    student = Submission(student_id="1", answers={"Q1": "$16", "Q2": "$16"})
    [marked] = mark(questions, scheme, [student], model_endpoint=DECIDING)
    assert (marked.results["Q1"].proportion, marked.results["Q2"].proportion) == (1, 0)


def test_a_mark_scheme_says_whether_it_needs_a_model_endpoint() -> None:
    assert SCHEME.needs_model_endpoint()
    plain = MarkScheme.load(
        [{"question": "Q1", "uses": "gradeflow/accepted-answers@1", "with": {"answers": ["16"]}}]
    )
    assert not plain.needs_model_endpoint()
    assert SCHEME.markings[0].needs_model_endpoint()
    assert not plain.markings[0].needs_model_endpoint()


def test_choosing_models_keeps_those_that_mark_the_examples_as_expected(
    fake_models: Install,
) -> None:
    def oracle(model: str, answer: str, criterion: str) -> Verdict:
        return "yes" if model == "careless" else examples_oracle(model, answer, criterion)

    fake = fake_models(oracle)
    fake.models = ["careless", "decider", "reasoner"]
    fake.without_logprobs = {"reasoner"}
    choice = choose_models(ENDPOINT)
    assert choice.endpoint.decision_model == "decider"
    assert choice.endpoint.reasoning_model in {"decider", "reasoner"}
    tried = [(t.model, t.role, t.matched, t.total, t.failure) for t in choice.trials]
    assert tried == [
        ("careless", "decision", 0, 9, "Missed 9 of 9 examples."),
        ("decider", "decision", 9, 9, None),
        ("reasoner", "decision", 0, 0, "Doesn't say how sure it is."),
        ("careless", "reasoning", 0, 9, "Missed 9 of 9 examples."),
        ("decider", "reasoning", 9, 9, None),
        ("reasoner", "reasoning", 9, 9, None),
    ]
    assert all(t.seconds >= 0 for t in choice.trials)


def test_choosing_models_says_why_a_model_fails(fake_models: Install) -> None:
    fake = fake_models(examples_oracle)
    fake.models = ["decider"]
    fake.listed = ["gone", "decider"]
    choice = choose_models(ENDPOINT)
    assert [t.failure for t in choice.trials if t.model == "gone"] == [
        "The AI service has no such model."
    ] * 2


def test_choosing_models_at_an_address_with_none_says_so(fake_models: Install) -> None:
    fake_models(examples_oracle)
    with pytest.raises(GradeFlowError, match="No models were found at this address"):
        choose_models(ENDPOINT.model_copy(update={"url": "https://models.test/wrong"}))


def test_listing_models_gives_the_endpoints_models(fake_models: Install) -> None:
    fake_models(examples_oracle).listed = ["decider", "reasoner"]
    assert list_models(ENDPOINT) == ["decider", "reasoner"]


def test_trying_one_model_says_how_it_did(fake_models: Install) -> None:
    fake = fake_models(examples_oracle)
    fake.without_logprobs = {"reasoner"}
    decider = try_model(ENDPOINT, "decision", "decider")
    assert (decider.matched, decider.total, decider.failure) == (9, 9, None)
    assert try_model(ENDPOINT, "decision", "reasoner").failure == "Doesn't say how sure it is."
