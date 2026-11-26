"""Language models a person brings: any OpenAI-compatible endpoint and its token."""

import asyncio
import itertools
import math
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ..common.exceptions import GradeFlowError

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
    from pydantic_ai.exceptions import ModelAPIError
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

TIMEOUT_SECONDS = 600
MAX_RETRIES = 5
"""Rate limits are the usual failure; the client waits as the service says before retrying."""
LIST_TIMEOUT_SECONDS = 15
LIST_MAX_RETRIES = 1
"""A person waits while the models are listed."""


class ServiceError(GradeFlowError):
    """A model service is unavailable or refused a request."""

    def __init__(self, message: str, *, fails_every_request: bool = False) -> None:
        super().__init__(message)
        self.fails_every_request = fails_every_request
        """Whether asking the model again would fail the same way, as with a wrong token."""


class ModelEndpoint(BaseModel):
    """Where a person's language models are, the model chosen for each role, and how many
    students' answers one request holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    api_key: SecretStr
    fast_model: str | None = None
    careful_model: str | None = None
    draft_model: str | None = None
    answers_per_request: int = Field(default=12, ge=1, le=40)


_MISSING_EXTRA = (
    "Using language-model endpoints requires gradeflow-core[llm]. Install it to use this feature."
)


def _provider(endpoint: ModelEndpoint) -> OpenAIProvider:
    try:
        import openai
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as error:
        raise ServiceError(_MISSING_EXTRA) from error
    # An endpoint can't send requests on to another address, inside a network it would otherwise
    # reach only through the endpoint.
    return OpenAIProvider(
        base_url=endpoint.url,
        api_key=endpoint.api_key.get_secret_value(),
        http_client=openai.DefaultAsyncHttpxClient(follow_redirects=False),
    )


def chat_model(endpoint: ModelEndpoint, model: str) -> OpenAIChatModel:
    """``model`` at ``endpoint``, for PydanticAI."""
    provider = _provider(endpoint)
    from pydantic_ai.models.openai import OpenAIChatModel

    return OpenAIChatModel(
        model,
        provider=provider,
        settings={"timeout": TIMEOUT_SECONDS, "max_retries": MAX_RETRIES},
    )


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
    """What one reply from ``model`` says, at temperature 0, without its reasoning."""
    response = _complete(
        endpoint,
        model,
        system_prompt,
        prompt,
        reasoning_effort=reasoning_effort,
        logprobs=False,
        max_tokens=max_tokens,
    )
    return (response.text or "").strip()


def ask_token_logprobs(
    endpoint: ModelEndpoint, model: str, system_prompt: str, prompt: str, *, max_tokens: int
) -> list[ChatCompletionTokenLogprob]:
    """The tokens of one reply from ``model``, at temperature 0 without reasoning, each with its
    likeliest alternatives."""
    response = _complete(
        endpoint,
        model,
        system_prompt,
        prompt,
        reasoning_effort="none",
        logprobs=True,
        max_tokens=max_tokens,
    )
    from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

    tokens = (response.provider_details or {}).get("logprobs", [])
    return [ChatCompletionTokenLogprob.model_validate(token) for token in tokens]


def _complete(
    endpoint: ModelEndpoint,
    model: str,
    system_prompt: str,
    prompt: str,
    *,
    reasoning_effort: ReasoningEffort | None,
    logprobs: bool,
    max_tokens: int,
) -> ModelResponse:
    openai_model = chat_model(endpoint, model)
    from pydantic_ai.direct import model_request
    from pydantic_ai.exceptions import ModelAPIError, ModelRequestRejectedError
    from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

    reasoning_effort = (
        None if (endpoint.url, model) in _MODELS_REFUSING_EFFORT else reasoning_effort
    )
    settings: OpenAIChatModelSettings = {"temperature": 0, "max_tokens": max_tokens}
    if logprobs:
        settings |= {"openai_logprobs": True, "openai_top_logprobs": 10}
    if reasoning_effort is not None:
        # PydanticAI leaves out an effort for models it doesn't know.
        settings |= {"extra_body": {"reasoning_effort": reasoning_effort}}

    async def send_request() -> ModelResponse:
        async with openai_model.client:
            return await model_request(
                openai_model,
                [ModelRequest([SystemPromptPart(system_prompt), UserPromptPart(prompt)])],
                model_settings=settings,
            )

    try:
        return asyncio.run(send_request())
    except ModelRequestRejectedError as error:
        if reasoning_effort is None or "reasoning_effort" not in str(error):
            raise service_error(error) from None
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
    except ModelAPIError as error:
        raise service_error(error) from None


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
    client = _provider(endpoint).client.with_options(
        timeout=LIST_TIMEOUT_SECONDS, max_retries=LIST_MAX_RETRIES
    )
    import openai

    async def fetch_models() -> list[openai.types.Model]:
        async with client:
            return [model async for model in client.models.list()]

    try:
        models = asyncio.run(fetch_models())
    except openai.NotFoundError:
        raise ServiceError(
            "No models were found at this address. Check it: it often ends in /v1."
        ) from None
    except openai.APIStatusError as error:
        raise _error_for_status(error.status_code) from None
    except openai.OpenAIError:
        raise _error_for_status(None) from None
    if any(model.created for model in models):
        models.sort(key=lambda model: model.created, reverse=True)
    return [model.id for model in models]


def service_error(error: ModelAPIError) -> ServiceError:
    from pydantic_ai.exceptions import ModelHTTPError

    return _error_for_status(error.status_code if isinstance(error, ModelHTTPError) else None)


def _error_for_status(status_code: int | None) -> ServiceError:
    """Why a request failed, as a sentence a lecturer can act on. ``None`` means the service
    wasn't reached."""
    match status_code:
        case None:
            message = "Couldn't reach the AI service."
        case 401 | 403:
            message = "The AI service refused the token."
        case 404:
            message = "The AI service has no such model."
        case 429:
            message = "The AI service is limiting requests."
        case _:
            message = f"The AI service answered {status_code}."
    return ServiceError(message, fails_every_request=status_code in (None, 401, 403, 404))
