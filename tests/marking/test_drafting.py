from collections.abc import Iterator
from typing import Any

import pytest

from gradeflow_core import (
    GradeFlowError,
    MarkScheme,
    QuestionSet,
    Submission,
    draft_marking,
)
from gradeflow_core.models.model_service import ServiceError
from tests.fake_models import ENDPOINT, FakeModels, InstallFakeModels, example_verdicts

DRAFT_ENDPOINT = ENDPOINT.model_copy(update={"draft_model": "drafter"})
QUESTIONS = QuestionSet.load(
    {"Q1": {"type": "text", "description": "What is the capital of France?"}}
)
SHEETS = [
    Submission(student_id=str(index), answers={"Q1": answer})
    for index, answer in enumerate(["Paris", "Paris", "paris ", "Lyon", ""])
]
PARIS = {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["Paris"]}}
PARIS_OR_LYON = {"uses": "gradeflow/accepted-answers@1", "with": {"answers": ["Paris", "Lyon"]}}
CUSTOM_CODE = {"uses": "gradeflow/custom-code@1", "with": {"code": "proportion = 1"}}
PARIS_AND_1889 = {
    "uses": "gradeflow/part-by-part@1",
    "with": {
        "checks": [
            {
                "input": "{{ answer.city }}",
                "uses": "gradeflow/accepted-answers@1",
                "with": {"answers": ["Paris"]},
            },
            {
                "input": "{{ answer.year }}",
                "uses": "gradeflow/number-range@1",
                "with": {"min_value": 1889, "max_value": 1889},
            },
        ]
    },
}


def script_replies(fake: FakeModels, *markings: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Replies with each marking in turn, then the last again; returns each request's messages."""
    replies: Iterator[dict[str, Any]] = iter(markings)
    conversations: list[list[dict[str, Any]]] = []
    last: dict[str, Any] = {}

    def draft(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal last
        conversations.append(messages)
        last = next(replies, last)
        return {"marking": last, "summary": f"  Accepts {last['with'].get('answers')}.\n"}

    fake.models.append("drafter")
    fake.drafts = draft
    return conversations


def draft(**options: Any) -> Any:
    return draft_marking(
        QUESTIONS, "Q1", "1 mark for Paris.", SHEETS, model_endpoint=DRAFT_ENDPOINT, **options
    )


def test_drafting_writes_a_marking_and_stops_once_the_model_keeps_it(
    fake_models: InstallFakeModels,
) -> None:
    conversations = script_replies(fake_models(example_verdicts), PARIS)
    stages: list[str] = []
    drafted = draft(on_stage=stages.append)
    assert drafted.marking.dump() == {"question": "Q1", **PARIS}
    assert drafted.summary == "Accepts ['Paris']."
    assert stages == ["Drafting the marking", "Checking it on 5 answers", "Revising the marking"]
    assert len(conversations) == 2


def test_the_model_reads_the_instructions_the_methods_and_the_normalized_answers(
    fake_models: InstallFakeModels,
) -> None:
    conversations = script_replies(fake_models(example_verdicts), PARIS)
    draft()
    task = conversations[0][-1]["content"]
    assert "1 mark for Paris." in task
    assert "What is the capital of France?" in task
    assert "(`gradeflow/accepted-answers@1`)" in task
    assert "gradeflow/custom-code@1" not in task, "a last resort, offered after a review"
    assert "gradeflow/answer-criteria@1" not in task, "asks a language model itself"
    assert '| "paris" | 3 |' in task, "lowercased, as the question reads it"
    assert "| (blank) | 1 |" in task
    review = conversations[1][-1]["content"]
    assert '| "lyon" | 1 | 0 |' in review


def test_a_marking_with_problems_goes_back_to_the_model(fake_models: InstallFakeModels) -> None:
    unknown_setting = {"uses": "gradeflow/accepted-answers@1", "with": {"answer": "Paris"}}
    conversations = script_replies(fake_models(example_verdicts), unknown_setting, PARIS)
    assert draft().marking.dump() == {"question": "Q1", **PARIS}
    assert "The marking has problems:\n- /with" in conversations[1][-1]["content"]


def test_custom_code_is_refused_until_a_review_offers_it(fake_models: InstallFakeModels) -> None:
    conversations = script_replies(fake_models(example_verdicts), CUSTOM_CODE, PARIS, CUSTOM_CODE)
    assert draft().marking.dump() == {"question": "Q1", **CUSTOM_CODE}
    assert "Custom Code isn't available. Use GradeFlow's methods." in str(conversations[1])
    assert "(`gradeflow/custom-code@1`)" in conversations[2][-1]["content"]


def test_the_model_revises_the_marking_from_its_review(fake_models: InstallFakeModels) -> None:
    conversations = script_replies(fake_models(example_verdicts), PARIS, PARIS_OR_LYON)
    drafted = draft()
    assert drafted.marking.dump() == {"question": "Q1", **PARIS_OR_LYON}
    assert drafted.summary == "Accepts ['Paris', 'Lyon']."
    assert len(conversations) == 3


def test_updating_starts_from_the_current_marking(fake_models: InstallFakeModels) -> None:
    conversations = script_replies(fake_models(example_verdicts), PARIS_OR_LYON)
    current = MarkScheme.load([{"question": "Q1", **PARIS}]).markings[0]
    stages: list[str] = []
    drafted = draft(marking=current, on_stage=stages.append)
    assert drafted.marking.dump() == {"question": "Q1", **PARIS_OR_LYON}
    assert stages[0] == "Updating the marking"
    task = conversations[0][-1]["content"]
    assert '# The current marking\n\n{"uses": "gradeflow/accepted-answers@1"' in task


def test_a_current_marking_of_custom_code_keeps_it_on_offer(
    fake_models: InstallFakeModels,
) -> None:
    conversations = script_replies(fake_models(example_verdicts), CUSTOM_CODE)
    current = MarkScheme.load([{"question": "Q1", **CUSTOM_CODE}]).markings[0]
    assert draft(marking=current).marking.dump() == {"question": "Q1", **CUSTOM_CODE}
    assert "(`gradeflow/custom-code@1`)" in conversations[0][-1]["content"]


def test_drafting_fails_when_ai_writes_no_usable_marking(
    fake_models: InstallFakeModels,
) -> None:
    script_replies(fake_models(example_verdicts), {"uses": "gradeflow/nothing@1", "with": {}})
    with pytest.raises(GradeFlowError, match=r"AI wrote no usable marking\."):
        draft()


def test_drafting_needs_a_draft_model() -> None:
    with pytest.raises(GradeFlowError, match="Choose a draft model"):
        draft_marking(QUESTIONS, "Q1", "1 mark for Paris.", SHEETS, model_endpoint=ENDPOINT)


def test_a_service_failure_says_why(fake_models: InstallFakeModels) -> None:
    fake_models(example_verdicts)
    with pytest.raises(ServiceError, match=r"^The AI service has no such model\.$"):
        draft()


def test_answers_the_question_cant_read_are_listed_apart(fake_models: InstallFakeModels) -> None:
    conversations = script_replies(fake_models(example_verdicts), PARIS_AND_1889)
    questions = QuestionSet.load(
        {
            "Q1": {
                "type": "multipart",
                "parts": [{"name": "city"}, {"name": "year", "type": "number"}],
            }
        }
    )
    sheets = [
        Submission(student_id=str(index), answers={"Q1": answer})
        for index, answer in enumerate(["Paris, 1889", "Paris"])
    ]
    draft_marking(questions, "Q1", "Paris and 1889.", sheets, model_endpoint=DRAFT_ENDPOINT)
    review = conversations[1][-1]["content"]
    assert "No marking marks them, Custom Code included; a person does." in review
    assert '| "Paris" | 1 |' in review


def test_a_revision_with_problems_keeps_the_marking_before_it(
    fake_models: InstallFakeModels,
) -> None:
    script_replies(
        fake_models(example_verdicts), PARIS, {"uses": "gradeflow/nothing@1", "with": {}}
    )
    assert draft().marking.dump() == {"question": "Q1", **PARIS}


def test_the_draft_model_is_not_asked_for_json_mode(fake_models: InstallFakeModels) -> None:
    fake = fake_models(example_verdicts)
    script_replies(fake, PARIS)
    draft()
    assert all("response_format" not in request for request in fake.requests)
