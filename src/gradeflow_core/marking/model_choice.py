"""Trying an endpoint's models on Answer Criteria's own examples, and the best of them so far."""

import time
from collections.abc import Callable, Iterable, Iterator
from typing import get_args

from pydantic import BaseModel

from ..common import model_service
from ..methods.catalogue import BUILTINS
from ..methods.text.answer_criteria import Role
from ..questions.question_set import QuestionSet
from .examples import mark_rows

ANSWER_CRITERIA = "gradeflow/answer-criteria@1"
ROLES: tuple[Role, ...] = get_args(Role)
PASSING_MODELS_PER_ROLE = 2
"""Models that must pass in a role before trying stops."""


class ModelTrial(BaseModel):
    """How one model did in one role: example rows marked as declared, of how many, and how long it
    took. ``failure`` says why it can't take the role; None when it marked every row."""

    model: str
    role: Role
    rows_matched: int
    rows_tried: int | None
    """None if the service failed."""
    seconds: float
    failure: str | None


def try_models(
    endpoint: model_service.ModelEndpoint,
    tried: Iterable[ModelTrial] = (),
    on_start: Callable[[str, Role], None] | None = None,
) -> Iterator[ModelTrial]:
    """Each trial as it ends: the endpoint's models, newest first, as the fast model until two
    pass, then as the careful model until two pass. A model already ``tried`` in a role counts
    without being tried again, unless the service failed on it. ``on_start`` hears of each trial as
    it starts. Raises ``GradeFlowError`` when the endpoint lists no models."""
    earlier = {(trial.model, trial.role): trial for trial in tried if trial.rows_tried is not None}
    model_names = model_service.list_models(endpoint)
    for role in ROLES:
        passed = 0
        for name in model_names:
            if passed == PASSING_MODELS_PER_ROLE:
                break
            trial = earlier.get((name, role))
            if trial is None:
                if on_start is not None:
                    on_start(name, role)
                trial = try_model(endpoint, role, name)
                yield trial
            passed += trial.failure is None


def best_models(trials: Iterable[ModelTrial]) -> tuple[str | None, str | None]:
    """The fast and the careful model: in each role, the one that marked the most example rows as
    declared, then the fastest; None where none marked any."""
    marked = [trial for trial in trials if trial.rows_tried]

    def best(role: Role) -> str | None:
        in_role = [trial for trial in marked if trial.role == role]
        top = min(in_role, key=lambda trial: (-trial.rows_matched, trial.seconds), default=None)
        return top.model if top else None

    return best("fast"), best("careful")


def try_model(endpoint: model_service.ModelEndpoint, role: Role, name: str) -> ModelTrial:
    """How ``name`` does as the ``role`` model on Answer Criteria's examples."""
    start = time.monotonic()
    trial_endpoint = endpoint.model_copy(update={"fast_model": name, "careful_model": name})

    def trial(rows_matched: int, rows_tried: int | None, failure: str | None) -> ModelTrial:
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
        return trial(0, None, str(error))
    method = BUILTINS.find(ANSWER_CRITERIA)
    rows_matched = rows_tried = 0
    for example in method.examples:
        questions = QuestionSet(example.questions)
        rows = mark_rows(
            ANSWER_CRITERIA,
            questions=questions,
            question=next(iter(questions)),
            settings={**example.with_, "decide_with": role},
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
