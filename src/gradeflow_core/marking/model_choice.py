"""Choosing an endpoint's models: those that mark Answer Criteria's own examples as declared."""

import time

from pydantic import BaseModel

from ..common import model_service
from ..methods.catalogue import BUILTINS
from ..methods.text.answer_criterion import Role
from ..questions.question_set import QuestionSet
from .examples import mark_rows

ANSWER_CRITERIA = "gradeflow/answer-criteria@1"
PASSING_MODELS_TO_COMPARE = 2
"""Models to try past the first that passes, for the faster of them."""


class ModelTrial(BaseModel):
    """How one model did in one role: example rows marked as declared, of how many, and how long it
    took. ``failure`` says why it can't take the role; None when it marked every row."""

    model: str
    role: Role
    rows_matched: int
    rows_tried: int
    seconds: float
    failure: str | None


class ModelChoice(BaseModel):
    endpoint: model_service.ModelEndpoint
    """The endpoint with its chosen models; a role no model passed stays None."""
    trials: list[ModelTrial]
    """Every model tried, in each role it was tried for, in order."""


def choose_models(endpoint: model_service.ModelEndpoint) -> ModelChoice:
    """For each role, the faster of the first models the endpoint lists, newest first, that mark
    every row of Answer Criteria's examples as they declare. Raises ``GradeFlowError`` when the
    endpoint lists no models."""
    model_names = model_service.list_models(endpoint)
    trials: list[ModelTrial] = []
    chosen_models: dict[Role, str | None] = {}
    for role in ("fast", "careful"):
        passing_trials: list[ModelTrial] = []
        for name in model_names:
            if len(passing_trials) == PASSING_MODELS_TO_COMPARE:
                break
            trials.append(trial := try_model(endpoint, role, name))
            if trial.failure is None:
                passing_trials.append(trial)
        chosen_models[role] = (
            min(passing_trials, key=lambda trial: trial.seconds).model if passing_trials else None
        )
    return ModelChoice(
        endpoint=endpoint.model_copy(
            update={"fast_model": chosen_models["fast"], "careful_model": chosen_models["careful"]}
        ),
        trials=trials,
    )


def try_model(endpoint: model_service.ModelEndpoint, role: Role, name: str) -> ModelTrial:
    """How ``name`` does as the ``role`` model on Answer Criteria's examples."""
    start = time.monotonic()
    trial_endpoint = endpoint.model_copy(update={"fast_model": name, "careful_model": name})

    def trial(rows_matched: int, rows_tried: int, failure: str | None) -> ModelTrial:
        seconds = round(time.monotonic() - start, 1)
        return ModelTrial(
            model=name,
            role=role,
            rows_matched=rows_matched,
            rows_tried=rows_tried,
            seconds=seconds,
            failure=failure,
        )

    try:
        if (probe_failure := _role_refusal(trial_endpoint, role, name)) is not None:
            return trial(0, 0, probe_failure)
    except model_service.ServiceError as error:
        return trial(0, 0, str(error))
    method = BUILTINS.find(ANSWER_CRITERIA)
    rows_matched = rows_tried = 0
    for example in method.examples:
        questions = QuestionSet(example.questions)
        rows = mark_rows(
            ANSWER_CRITERIA,
            questions=questions,
            question=next(iter(questions)),
            inputs={**example.with_, "decide_with": role},
            rows=example.rows,
            model_endpoint=trial_endpoint,
        )
        rows_tried += len(rows)
        rows_matched += sum(bool(row.matches_expected) for row in rows)
    missed = rows_tried - rows_matched
    return trial(
        rows_matched,
        rows_tried,
        f"Missed {missed} of {rows_tried} example answers." if missed else None,
    )


def _role_refusal(endpoint: model_service.ModelEndpoint, role: Role, name: str) -> str | None:
    """Why one plain question shows ``name`` can't be the ``role`` model, or None."""
    system_prompt, question = "Answer with one word: Yes or No.", "Is 2 + 2 equal to 4?"
    if role == "fast":
        reply = model_service.ask(
            endpoint, name, system_prompt, question, effort="none", logprobs=True, max_tokens=4
        )
        return (
            None
            if model_service.yes_probabilities(reply)
            else "Doesn't return token probabilities."
        )
    reply = model_service.ask(endpoint, name, system_prompt, question, effort=None, max_tokens=4000)
    return (
        None
        if "yes" in model_service.reply_text(reply).lower()
        else "Didn't answer a simple question."
    )
