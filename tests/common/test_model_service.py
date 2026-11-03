from functools import partial

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion

from gradeflow_core import GradeFlowError, ModelEndpoint
from gradeflow_core.common import model_service
from tests.fake_models import ENDPOINT


def completion(top_logprobs: list[dict[str, object]]) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Yes"},
                    "logprobs": {
                        "content": [{"token": "Yes", "logprob": -0.1, "top_logprobs": top_logprobs}]
                    },
                }
            ],
        }
    )


def test_yes_probability_weighs_yes_against_no() -> None:
    top_logprobs = [{"token": " Yes", "logprob": -0.2231}, {"token": "No", "logprob": -1.6094}]
    [probability] = model_service.yes_probabilities(completion(top_logprobs))
    assert round(probability, 2) == 0.8


def test_yes_probability_without_yes_or_no_among_the_alternatives_is_the_word_written() -> None:
    assert model_service.yes_probabilities(completion([{"token": "Maybe", "logprob": -0.1}])) == [
        1.0
    ]


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

    with openai.OpenAI(
        base_url=ENDPOINT.url,
        api_key=ENDPOINT.token.get_secret_value(),
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:

        def fake_client(endpoint: ModelEndpoint) -> openai.OpenAI:
            return client

        monkeypatch.setattr(model_service, "_client", fake_client)
        call = (
            partial(
                model_service.ask, ENDPOINT, "model", "system", "prompt", effort=None, max_tokens=4
            )
            if operation == "ask"
            else partial(model_service.list_models, ENDPOINT)
        )
        with pytest.raises(GradeFlowError) as caught:
            call()
    if operation == "list_models" and status == 404:
        message = "No models were found at this address. Check it: it often ends in /v1."
    assert str(caught.value) == message
