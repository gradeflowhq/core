"""Language models a person brings: any OpenAI-compatible endpoint and its token."""

import math
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .exceptions import GradeFlowError

if TYPE_CHECKING:
    import openai
    from openai.types.chat import ChatCompletion

TIMEOUT = 600
RETRIES = 5
"""Rate limits are the usual failure; the client waits as the service says before retrying."""
_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class ServiceError(GradeFlowError):
    """A model service is unavailable or refused a request."""


class ModelEndpoint(BaseModel):
    """Where a person's language models are, the model chosen for each role, and how many
    students' answers one request holds."""

    model_config = ConfigDict(frozen=True)

    url: str
    token: SecretStr
    decision_model: str | None = None
    reasoning_model: str | None = None
    answers_per_request: int = Field(default=12, ge=1, le=40)


@lru_cache(maxsize=8)
def _client(endpoint: ModelEndpoint) -> openai.OpenAI:
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
        api_key=endpoint.token.get_secret_value(),
        timeout=TIMEOUT,
        max_retries=RETRIES,
        http_client=openai.DefaultHttpxClient(follow_redirects=False),
    )


Effort = Literal["none", "low", "medium", "high"]
_NO_EFFORT: set[tuple[str, str]] = set()
"""Endpoints' models that refuse a reasoning effort, asked without one after the first refusal."""


def ask(
    endpoint: ModelEndpoint,
    model: str,
    system: str,
    prompt: str,
    *,
    effort: Effort | None,
    logprobs: bool = False,
    max_tokens: int,
) -> ChatCompletion:
    """One reply from ``model``, at temperature 0."""
    client = _client(endpoint)
    import openai

    effort = None if (endpoint.url, model) in _NO_EFFORT else effort
    try:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            logprobs=True if logprobs else openai.omit,
            top_logprobs=10 if logprobs else openai.omit,
            reasoning_effort=openai.omit if effort is None else effort,
        )
    except openai.BadRequestError as error:
        if effort is None or "reasoning_effort" not in str(error):
            raise ServiceError(failure(error)) from None
        _NO_EFFORT.add((endpoint.url, model))
        return ask(
            endpoint, model, system, prompt, effort=None, logprobs=logprobs, max_tokens=max_tokens
        )
    except openai.OpenAIError as error:
        raise ServiceError(failure(error)) from None


def yes_probabilities(reply: ChatCompletion) -> list[float]:
    """P(yes) at each Yes or No the reply wrote, in order, from the logprobs of the tokens it
    could have written there."""
    logprobs = reply.choices[0].logprobs
    found = []
    for token in (logprobs.content or []) if logprobs else []:
        if token.token.strip().lower() not in ("yes", "no"):
            continue
        mass = {"yes": 0.0, "no": 0.0}
        for option in token.top_logprobs:
            word = option.token.strip().lower()
            if word in mass:
                mass[word] += math.exp(option.logprob)
        total = mass["yes"] + mass["no"]
        found.append(mass["yes"] / total if total else float(token.token.strip().lower() == "yes"))
    return found


def text(reply: ChatCompletion) -> str:
    """What the reply says, without reasoning a model wrote inline."""
    return _THINKING.sub("", reply.choices[0].message.content or "").strip()


def model_names(endpoint: ModelEndpoint) -> list[str]:
    """The endpoint's models, newest first when it says when each was made."""
    client = _client(endpoint)
    import openai

    try:
        listed = list(client.models.list())
    except openai.NotFoundError:
        raise ServiceError(
            "No models were found at this address. Check it: it often ends in /v1."
        ) from None
    except openai.OpenAIError as error:
        raise ServiceError(failure(error)) from None
    if any(model.created for model in listed):
        listed.sort(key=lambda model: model.created, reverse=True)
    return [model.id for model in listed]


def failure(error: openai.OpenAIError) -> str:
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
