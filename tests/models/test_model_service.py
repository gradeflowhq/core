from functools import partial

import httpx2
import openai
import pytest
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from pydantic import ValidationError

from gradeflow_core import GradeFlowError
from gradeflow_core.models import model_service
from tests.fake_models import ENDPOINT, install_service


def tokens(top_logprobs: list[dict[str, object]]) -> list[ChatCompletionTokenLogprob]:
    token = {"token": "Yes", "logprob": -0.1, "top_logprobs": top_logprobs}
    return [ChatCompletionTokenLogprob.model_validate(token)]


def test_yes_probability_weighs_yes_against_no() -> None:
    top_logprobs = [{"token": " Yes", "logprob": -0.2231}, {"token": "No", "logprob": -1.6094}]
    [probability] = model_service.yes_probabilities(tokens(top_logprobs))
    assert round(probability, 2) == 0.8


def test_yes_probability_without_yes_or_no_among_the_alternatives_is_the_word_written() -> None:
    assert model_service.yes_probabilities(tokens([{"token": "Maybe", "logprob": -0.1}])) == [1.0]


@pytest.mark.parametrize("operation", ["ask", "list_models"])
@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, "The AI service answered 400."),
        (401, "The AI service refused the token."),
        (403, "The AI service refused the token."),
        (404, "The AI service has no such model."),
        (429, "The AI service is limiting requests."),
        (500, "The AI service answered 500."),
    ],
)
def test_service_failures_use_domain_errors(
    monkeypatch: pytest.MonkeyPatch, operation: str, status: int, message: str
) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "Refused."}})

    install_service(monkeypatch, respond)
    call = (
        partial(
            model_service.ask,
            ENDPOINT,
            "model",
            "system",
            "prompt",
            reasoning_effort=None,
            max_tokens=4,
        )
        if operation == "ask"
        else partial(model_service.list_models, ENDPOINT)
    )
    with pytest.raises(GradeFlowError) as exc_info:
        call()
    if operation == "list_models" and status == 404:
        message = "No models were found at this address. Check it: it often ends in /v1."
    assert str(exc_info.value) == message


def test_listing_models_waits_briefly_retries_once_and_closes_its_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        status = 500 if len(requests) == 1 else 200
        return httpx2.Response(status, json={"object": "list", "data": []})

    clients: list[httpx2.AsyncClient] = []

    def client(**_: object) -> httpx2.AsyncClient:
        clients.append(httpx2.AsyncClient(transport=httpx2.MockTransport(respond)))
        return clients[-1]

    monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", client)
    assert model_service.list_models(ENDPOINT) == []
    assert len(requests) == 2
    assert requests[0].extensions["timeout"]["read"] == 15
    assert all(listing_client.is_closed for listing_client in clients)


def test_an_endpoint_refuses_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        model_service.ModelEndpoint.model_validate(
            {"url": "https://models.example", "api_key": "key", "fast_modle": "m"}
        )
