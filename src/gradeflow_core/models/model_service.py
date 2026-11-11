"""Language models a person brings: any OpenAI-compatible endpoint and its token."""

import itertools
import math
import re
from collections.abc import Collection, Iterable
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ..common.exceptions import GradeFlowError

if TYPE_CHECKING:
    import openai
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

TIMEOUT_SECONDS = 600
MAX_RETRIES = 5
"""Rate limits are the usual failure; the client waits as the service says before retrying."""
LIST_TIMEOUT_SECONDS = 15
LIST_MAX_RETRIES = 1
"""A person waits while the models are listed."""
_THINKING_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class ServiceError(GradeFlowError):
    """A model service is unavailable or refused a request."""

    def __init__(self, message: str, *, fails_every_request: bool = False) -> None:
        super().__init__(message)
        self.fails_every_request = fails_every_request
        """Whether asking the model again would fail the same way, as with a wrong token."""


class ModelEndpoint(BaseModel):
    """Where a person's language models are, the model chosen for each role, and how many
    students' answers one request holds."""

    model_config = ConfigDict(frozen=True)

    url: str
    api_key: SecretStr
    fast_model: str | None = None
    careful_model: str | None = None
    answers_per_request: int = Field(default=12, ge=1, le=40)


def _client(
    endpoint: ModelEndpoint, *, timeout: float = TIMEOUT_SECONDS, max_retries: int = MAX_RETRIES
) -> openai.OpenAI:
    try:
        import openai
    except ImportError as error:
        raise ServiceError(
            "Using language-model endpoints requires gradeflow-core[llm]. "
            "Install it to use this feature."
        ) from error
    # An endpoint can't send requests on to another address, inside a network it would otherwise
    # reach only through the endpoint.
    return openai.OpenAI(
        base_url=endpoint.url,
        api_key=endpoint.api_key.get_secret_value(),
        timeout=timeout,
        max_retries=max_retries,
        http_client=openai.DefaultHttpxClient(follow_redirects=False),
    )


_cached_client = lru_cache(maxsize=8)(_client)


ReasoningEffort = Literal["none", "low", "medium", "high"]
_MODELS_REFUSING_EFFORT: set[tuple[str, str]] = set()
"""Endpoints' models that refuse a reasoning effort, asked without one after the first refusal."""


def ask(
    endpoint: ModelEndpoint,
    model: str,
    system_prompt: str,
    prompt: str,
    *,
    reasoning_effort: ReasoningEffort | None,
    max_tokens: int,
) -> str:
    """What one reply from ``model`` says, at temperature 0, without reasoning it wrote inline."""
    reply = _complete(
        endpoint,
        model,
        system_prompt,
        prompt,
        reasoning_effort=reasoning_effort,
        logprobs=False,
        max_tokens=max_tokens,
    )
    return _THINKING_BLOCK.sub("", reply.choices[0].message.content or "").strip()


def ask_token_logprobs(
    endpoint: ModelEndpoint, model: str, system_prompt: str, prompt: str, *, max_tokens: int
) -> list[ChatCompletionTokenLogprob]:
    """The tokens of one reply from ``model``, at temperature 0 without reasoning, each with its
    likeliest alternatives."""
    reply = _complete(
        endpoint,
        model,
        system_prompt,
        prompt,
        reasoning_effort="none",
        logprobs=True,
        max_tokens=max_tokens,
    )
    logprobs = reply.choices[0].logprobs
    return (logprobs.content or []) if logprobs else []


def _complete(
    endpoint: ModelEndpoint,
    model: str,
    system_prompt: str,
    prompt: str,
    *,
    reasoning_effort: ReasoningEffort | None,
    logprobs: bool,
    max_tokens: int,
) -> ChatCompletion:
    client = _cached_client(endpoint)
    import openai

    reasoning_effort = (
        None if (endpoint.url, model) in _MODELS_REFUSING_EFFORT else reasoning_effort
    )
    try:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            logprobs=True if logprobs else openai.omit,
            top_logprobs=10 if logprobs else openai.omit,
            reasoning_effort=openai.omit if reasoning_effort is None else reasoning_effort,
        )
    except openai.BadRequestError as error:
        if reasoning_effort is None or "reasoning_effort" not in str(error):
            raise ServiceError(failure_message(error)) from None
        _MODELS_REFUSING_EFFORT.add((endpoint.url, model))
        return _complete(
            endpoint,
            model,
            system_prompt,
            prompt,
            reasoning_effort=None,
            logprobs=logprobs,
            max_tokens=max_tokens,
        )
    except openai.OpenAIError as error:
        fails_every_request = isinstance(
            error,
            openai.APIConnectionError
            | openai.AuthenticationError
            | openai.PermissionDeniedError
            | openai.NotFoundError,
        )
        raise ServiceError(
            failure_message(error), fails_every_request=fails_every_request
        ) from None


def _normalized_probabilities(
    token: ChatCompletionTokenLogprob, options: Collection[str]
) -> dict[str, float]:
    """The probability of each of ``options`` at ``token``, normalized over its alternatives."""
    probabilities = dict.fromkeys(options, 0.0)
    for alternative in token.top_logprobs:
        option = alternative.token.strip().lower()
        if option in probabilities:
            probabilities[option] += math.exp(alternative.logprob)
    probability_sum = sum(probabilities.values())
    token_text = token.token.strip().lower()
    return {
        option: probabilities[option] / probability_sum
        if probability_sum
        else float(option == token_text)
        for option in options
    }


def yes_probabilities(tokens: Iterable[ChatCompletionTokenLogprob]) -> list[float]:
    """P(yes) at each Yes or No among ``tokens``, in order."""
    return [
        _normalized_probabilities(token, ("yes", "no"))["yes"]
        for token in tokens
        if token.token.strip().lower() in ("yes", "no")
    ]


def option_probabilities(
    tokens: Iterable[ChatCompletionTokenLogprob], options: Collection[str]
) -> list[tuple[str, float]]:
    """The option written after each colon among ``tokens``, and its probability."""
    non_blank_tokens = [token for token in tokens if token.token.strip()]
    return [
        (option, _normalized_probabilities(token, options)[option])
        for previous_token, token in itertools.pairwise(non_blank_tokens)
        if previous_token.token.strip().endswith(":")
        and (option := token.token.strip().lower()) in options
    ]


def list_models(endpoint: ModelEndpoint) -> list[str]:
    """The endpoint's models, newest first when it says when each was made."""
    client = _client(endpoint, timeout=LIST_TIMEOUT_SECONDS, max_retries=LIST_MAX_RETRIES)
    import openai

    try:
        with client:
            models = list(client.models.list())
    except openai.NotFoundError:
        raise ServiceError(
            "No models were found at this address. Check it: it often ends in /v1."
        ) from None
    except openai.OpenAIError as error:
        raise ServiceError(failure_message(error)) from None
    if any(model.created for model in models):
        models.sort(key=lambda model: model.created, reverse=True)
    return [model.id for model in models]


def failure_message(error: openai.OpenAIError) -> str:
    """Why a request failed, as a sentence a lecturer can act on."""
    import openai

    match error:
        case openai.AuthenticationError() | openai.PermissionDeniedError():
            return "The AI service refused the token."
        case openai.NotFoundError():
            return "The AI service has no such model."
        case openai.RateLimitError():
            return "The AI service is limiting requests."
        case openai.APIConnectionError():
            return "Couldn't reach the AI service."
        case openai.APIStatusError():
            return f"The AI service answered {error.status_code}."
    return "The AI service failed."
