"""Choosing an endpoint's models: those that mark Answer Criteria's own examples as declared."""

import time

from pydantic import BaseModel

from ..common import model_service
from ..methods.catalogue import BUILTINS
from ..methods.text.answer_criterion import Role
from ..questions.question_set import QuestionSet
from .examples import evaluate_rows

METHOD = "gradeflow/answer-criteria@1"
ENOUGH = 2
"""Models to try past the first that passes, for the faster of them."""


class ModelTrial(BaseModel):
    """How one model did in one role: example rows marked as declared, of how many, and how long it
    took. ``failure`` says why it can't take the role; None when it marked every row."""

    model: str
    role: Role
    matched: int
    total: int
    seconds: float
    failure: str | None


class ModelChoice(BaseModel):
    endpoint: model_service.ModelEndpoint
    """The endpoint with its chosen models; a role no model passed stays None."""
    trials: list[ModelTrial]
    """Every model tried, in each role it was tried for, in order."""


def list_models(endpoint: model_service.ModelEndpoint) -> list[str]:
    """The endpoint's models, newest first when it says when each was made. Raises
    ``GradeFlowError`` when it lists none."""
    return model_service.model_names(endpoint)


def choose_models(endpoint: model_service.ModelEndpoint) -> ModelChoice:
    """For each role, the faster of the first models the endpoint lists, newest first, that mark
    every row of Answer Criteria's examples as they declare. Raises ``GradeFlowError`` when the
    endpoint lists no models."""
    names = list_models(endpoint)
    trials: list[ModelTrial] = []
    chosen: dict[Role, str | None] = {}
    for role in ("decision", "reasoning"):
        passed: list[ModelTrial] = []
        for name in names:
            if len(passed) == ENOUGH:
                break
            trials.append(trial := try_model(endpoint, role, name))
            if trial.failure is None:
                passed.append(trial)
        chosen[role] = min(passed, key=lambda trial: trial.seconds).model if passed else None
    return ModelChoice(
        endpoint=endpoint.model_copy(
            update={"decision_model": chosen["decision"], "reasoning_model": chosen["reasoning"]}
        ),
        trials=trials,
    )


def try_model(endpoint: model_service.ModelEndpoint, role: Role, name: str) -> ModelTrial:
    """How ``name`` does as the ``role`` model on Answer Criteria's examples."""
    start = time.monotonic()
    trying = endpoint.model_copy(update={"decision_model": name, "reasoning_model": name})

    def done(matched: int, total: int, failure: str | None) -> ModelTrial:
        seconds = round(time.monotonic() - start, 1)
        return ModelTrial(
            model=name, role=role, matched=matched, total=total, seconds=seconds, failure=failure
        )

    try:
        if (refusal := _probe(trying, role, name)) is not None:
            return done(0, 0, refusal)
    except model_service.ServiceError as error:
        return done(0, 0, str(error))
    method = BUILTINS.find(METHOD)
    matched = total = 0
    for example in method.examples:
        questions = QuestionSet(example.questions)
        rows = evaluate_rows(
            METHOD,
            questions=questions,
            question=next(iter(questions)),
            inputs={**example.with_, "decide_with": role},
            rows=example.rows,
            model_endpoint=trying,
        )
        total += len(rows)
        matched += sum(bool(row.matches) for row in rows)
    missed = total - matched
    return done(matched, total, f"Missed {missed} of {total} examples." if missed else None)


def _probe(endpoint: model_service.ModelEndpoint, role: Role, name: str) -> str | None:
    """Why one plain question shows ``name`` can't be the ``role`` model, or None."""
    system, question = "Answer with one word: Yes or No.", "Is 2 + 2 equal to 4?"
    if role == "decision":
        reply = model_service.ask(
            endpoint, name, system, question, effort="none", logprobs=True, max_tokens=4
        )
        return None if model_service.yes_probabilities(reply) else "Doesn't say how sure it is."
    reply = model_service.ask(endpoint, name, system, question, effort=None, max_tokens=4000)
    return None if "yes" in model_service.text(reply).lower() else "Gave no answer."
