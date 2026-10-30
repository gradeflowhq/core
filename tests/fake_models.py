"""A model service for tests: an OpenAI-compatible endpoint answering from a table of verdicts."""

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx2
import openai
import pytest
from pydantic import SecretStr

from gradeflow_core import ModelEndpoint
from gradeflow_core.common import model_service

ENDPOINT = ModelEndpoint(url="https://models.test/v1", token=SecretStr("key"))

type Verdict = float | str
"""P(yes), or "yes", "no" or "unsure: reason". The decision model reads a word as 0.95 or 0.05;
the reasoning model reads a probability as yes from 0.5."""
type Oracle = Callable[[str, str, str], Verdict]
"""The verdict of a model, by model name, answer and criterion."""

_ITEM = re.compile(r"^(\d+)\. (.+)$", re.MULTILINE)
_QUESTION = re.compile(r"^Question: (.*)$", re.MULTILINE)
_ANSWER = re.compile(
    r'<answer id="(A\d+)">\n(.*?)\n</answer>(?:\nDecide items for A\d+: ([\d, ]+))?', re.DOTALL
)


@dataclass
class FakeModels:
    oracle: Oracle
    models: list[str] = field(default_factory=lambda: ["decider", "reasoner"])
    refuses_effort: set[str] = field(default_factory=set)
    """Models that answer 400 when asked for a reasoning effort."""
    listed: list[str] | None = None
    """What the endpoint lists, when that isn't the models it serves."""
    without_logprobs: set[str] = field(default_factory=set)
    """Models that reply without logprobs, as many reasoning models do."""
    failing: set[str] = field(default_factory=set)
    """Answers whose requests the service fails with a server error."""
    garbled: int = 0
    """How many replies to break before answering properly."""
    answers_per_reply: int | None = None
    """How many answers a model can mark in one reply; for more, its reply can't be read."""
    by_question: dict[str, Oracle] = field(default_factory=dict)
    """Oracles for prompts about a question, by its description."""
    requests: list[dict[str, Any]] = field(default_factory=list)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if not request.url.path.startswith("/v1/"):
            return httpx2.Response(404, json={"error": {"message": "Not found."}})
        if request.url.path.endswith("/models"):
            listed = [
                {"id": m, "object": "model", "created": 0, "owned_by": "x"}
                for m in self.listed or self.models
            ]
            return httpx2.Response(200, json={"object": "list", "data": listed})
        body = json.loads(request.content)
        self.requests.append(body)
        model, prompt = body["model"], body["messages"][-1]["content"]
        answers = _ANSWER.findall(prompt)
        if model not in self.models:
            return httpx2.Response(404, json={"error": {"message": f"No model {model}."}})
        if model in self.refuses_effort and "reasoning_effort" in body:
            message = "Unsupported parameter: reasoning_effort."
            return httpx2.Response(400, json={"error": {"message": message}})
        if self.answers_per_reply is not None and len(answers) > self.answers_per_reply:
            return _reply("I can't tell.", [])
        if any(text in self.failing for _, text, _ in answers):
            return httpx2.Response(500, json={"error": {"message": "Overloaded."}})
        logprobs = bool(body.get("logprobs")) and model not in self.without_logprobs
        if not answers:  # a plain question, as choosing models asks first
            return _reply("Yes", [_token("Yes", 0.99)] if logprobs else [])
        if self.garbled:
            self.garbled -= 1
            return _reply("I can't tell.", [])
        preamble = prompt.split("\n\n<answer")[0]
        items = dict(_ITEM.findall(preamble))
        asked = _QUESTION.match(preamble)
        oracle = self.by_question.get(asked[1], self.oracle) if asked else self.oracle
        if body.get("logprobs"):
            return self._decided(oracle, model, items, answers, logprobs)
        return self._reasoned(oracle, model, items, answers)

    def _decided(
        self,
        oracle: Oracle,
        model: str,
        items: dict[str, str],
        answers: list[tuple[str, str, str]],
        logprobs: bool,
    ) -> httpx2.Response:
        lines, tokens = [], []
        for label, text, _ in answers:
            lines.append(f"{label}:")
            tokens.append(_token(f"{label}:", 1.0))
            for number in sorted(items, key=int):
                p = _probability(oracle(model, text, items[number]))
                word = "Yes" if p >= 0.5 else "No"
                lines[-1] += f" {word}"
                tokens.append(_token(f" {word}", p))
        return _reply("\n".join(lines), tokens if logprobs else [])

    def _reasoned(
        self, oracle: Oracle, model: str, items: dict[str, str], answers: list[tuple[str, str, str]]
    ) -> httpx2.Response:
        lines = []
        for label, text, asked in answers:
            for number in (int(n) for n in asked.split(",")):
                word, _, reason = _word(oracle(model, text, items[str(number)])).partition(":")
                said = f" ({reason.strip()})" if reason else ""
                lines.append(f"{label} {number}: {word.capitalize()}{said}")
        return _reply("<think>Looking at each answer.</think>\n" + "\n".join(lines), [])


def _probability(verdict: Verdict) -> float:
    if isinstance(verdict, float | int):
        return float(verdict)
    return 0.95 if verdict == "yes" else 0.05


def _word(verdict: Verdict) -> str:
    if isinstance(verdict, str):
        return verdict
    return "yes" if verdict >= 0.5 else "no"


def examples_oracle(model: str, answer: str, criterion: str) -> Verdict:
    """A model that marks Answer Criteria's own examples as they declare."""
    return "yes" if criterion in EXAMPLE_TRUTH.get(answer, set()) else "no"


EXAMPLE_TRUTH = {
    "3 × 2 = 6 and 2 × 5 = 10, so 6 + 10 = $16": {"Correct total", "Shows the working"},
    "3 × 2 = 6 and 2 × 5 = 10, so she pays $6 and $10": {"Shows the working", "Forgot to sum"},
    "$16": {"Correct total"},
    "Water expands as it freezes, so ice is less dense than liquid water.": {
        "Says ice is less dense than water",
        "Explains that water expands as it freezes",
    },
    "Because ice is less dense than water.": {"Says ice is less dense than water"},
    "Ice is denser than water, so it floats.": {"Says ice is denser than water"},
    "def largest(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n"
    "            best = x\n    return best": {
        "Compares each number with the largest so far",
        "Works when every number is negative",
    },
    "def largest(xs):\n    best = 0\n    for x in xs:\n        if x > best:\n"
    "            best = x\n    return best": {"Compares each number with the largest so far"},
    "def largest(xs):\n    return max(xs)": {"Works when every number is negative", "Uses max"},
}
"""Which criteria fit each answer of Answer Criteria's examples."""


def _token(text: str, p: float) -> dict[str, Any]:
    yes, no = math.log(max(p, 1e-9)), math.log(max(1 - p, 1e-9))
    return {
        "token": text,
        "logprob": yes if p >= 0.5 else no,
        "bytes": None,
        "top_logprobs": [
            {"token": " Yes", "logprob": yes, "bytes": None},
            {"token": " No", "logprob": no, "bytes": None},
        ],
    }


def _reply(content: str, tokens: list[dict[str, Any]]) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                    "logprobs": {"content": tokens} if tokens else None,
                }
            ],
        },
    )


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> Callable[[Oracle], FakeModels]:
    """Point every model endpoint at a fake service answering from the given oracle."""

    def install(oracle: Oracle) -> FakeModels:
        fake = FakeModels(oracle)

        def client(endpoint: ModelEndpoint) -> openai.OpenAI:
            return openai.OpenAI(
                base_url=endpoint.url,
                api_key=endpoint.token.get_secret_value(),
                max_retries=0,
                http_client=httpx2.Client(transport=httpx2.MockTransport(fake.handle)),
            )

        monkeypatch.setattr(model_service, "_client", client)
        return fake

    return install
