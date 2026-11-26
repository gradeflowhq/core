"""Trying an endpoint's models in each role, and the best of them so far: fast and careful on
Answer Criteria's own examples, draft on the basic methods' examples."""

import time
from collections.abc import Callable, Iterable, Iterator
from math import isclose
from typing import Literal, get_args

from pydantic import BaseModel

from ..common.exceptions import GradeFlowError
from ..methods.catalogue import BUILTINS
from ..methods.text.answer_criteria import AnswerCriteria, DecisionRole, examples_decided_with
from ..models import model_service
from ..questions.question_set import QuestionSet
from ..steps.example import Example
from ..steps.names import method_ref
from ..steps.outcome import ROUNDING_TOLERANCE
from ..steps.step import valid_step
from .drafting import draft_marking
from .examples import mark_rows, method_step_data
from .mark import mark
from .mark_scheme import MarkScheme

ModelRole = Literal[DecisionRole, "draft"]
"""What a model is chosen for: Answer Criteria's fast or careful decisions, or drafting markings."""
ROLES: tuple[ModelRole, ...] = get_args(ModelRole)
PASSING_MODELS_PER_ROLE = 2
"""Models that must pass in a role before trying stops."""
DRAFT_TRIAL_TIME_LIMIT_SECONDS = 15 * 60
"""Seconds after which a draft trial starts no more examples."""


class ModelTrial(BaseModel):
    """How one model did in one role: example rows marked as declared, of how many, and how long it
    took. ``failure`` says why it can't take the role; None when it marked every row."""

    model: str
    role: ModelRole
    rows_matched: int
    rows_tried: int | None
    """None if the service failed."""
    seconds: float
    failure: str | None


def try_models(
    endpoint: model_service.ModelEndpoint,
    tried: Iterable[ModelTrial] = (),
    on_start: Callable[[str, ModelRole], None] | None = None,
) -> Iterator[ModelTrial]:
    """Each trial as it ends: the endpoint's models, newest first, as the fast model until two
    pass, then likewise as the careful and the draft model. A model already ``tried`` in a role
    counts without being tried again, unless the service failed on it. ``on_start`` hears of each
    trial as it starts. Raises ``GradeFlowError`` when the endpoint lists no models."""
    earlier_trials = {
        (trial.model, trial.role): trial for trial in tried if trial.rows_tried is not None
    }
    model_names = model_service.list_models(endpoint)
    for role in ROLES:
        passing_model_count = 0
        for name in model_names:
            if passing_model_count == PASSING_MODELS_PER_ROLE:
                break
            trial = earlier_trials.get((name, role))
            if trial is None:
                if on_start is not None:
                    on_start(name, role)
                trial = try_model(endpoint, role, name)
                yield trial
            passing_model_count += trial.failure is None


def best_models(trials: Iterable[ModelTrial]) -> tuple[str | None, str | None, str | None]:
    """The fast, the careful and the draft model: in each role, the one that marked the most
    example rows as declared, then the fastest; None where none marked any."""
    marked = [trial for trial in trials if trial.rows_tried]

    def best_model(role: ModelRole) -> str | None:
        role_trials = [trial for trial in marked if trial.role == role]
        best_trial = min(
            role_trials, key=lambda trial: (-trial.rows_matched, trial.seconds), default=None
        )
        return best_trial.model if best_trial else None

    return best_model("fast"), best_model("careful"), best_model("draft")


def try_model(endpoint: model_service.ModelEndpoint, role: ModelRole, name: str) -> ModelTrial:
    """How ``name`` does as the ``role`` model: on Answer Criteria's examples, or drafting markings
    of the basic methods' first examples from their descriptions."""
    start = time.monotonic()
    trial_endpoint = endpoint.model_copy(
        update={"fast_model": name, "careful_model": name, "draft_model": name}
    )

    def make_trial(rows_matched: int, rows_tried: int | None, failure: str | None) -> ModelTrial:
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
            return make_trial(0, 0, probe_failure)
        rows_matched, rows_tried = (
            _drafting_row_counts(trial_endpoint)
            if role == "draft"
            else _decision_row_counts(trial_endpoint, role)
        )
    except model_service.ServiceError as error:
        return make_trial(0, None, str(error))
    missed_row_count = rows_tried - rows_matched
    return make_trial(
        rows_matched,
        rows_tried,
        f"Missed {missed_row_count} of {rows_tried} example answers." if missed_row_count else None,
    )


def _decision_row_counts(
    endpoint: model_service.ModelEndpoint, role: DecisionRole
) -> tuple[int, int]:
    """How many of Answer Criteria's example rows the ``role`` model marks as declared, and how
    many were tried."""
    reference = method_ref(AnswerCriteria.name, AnswerCriteria.version)
    rows_matched = rows_tried = 0
    for example in examples_decided_with(role):
        questions = QuestionSet(example.questions)
        row_results = mark_rows(
            reference,
            questions=questions,
            question=next(iter(questions)),
            settings=example.with_,
            rows=example.rows,
            model_endpoint=endpoint,
        )
        rows_tried += len(row_results)
        rows_matched += sum(bool(row_result.matches_expected) for row_result in row_results)
    return rows_matched, rows_tried


def _drafting_examples() -> list[tuple[str, Example]]:
    """Each basic method that gives marks without a language model, with its first example."""
    return [
        (method.ref, method.examples[0])
        for method in BUILTINS.latest()
        if method.level == "basic"
        and not method.needs_model_endpoint
        and method.gives_marks(method.examples[0].with_)
    ]


def _drafting_row_counts(endpoint: model_service.ModelEndpoint) -> tuple[int, int]:
    """How many example rows the draft model's markings gave their declared marks, and how many
    were tried: each example's question, with its title as the description, and the description
    of its marking as the instructions. A service failure ends the trial; an example it couldn't
    draft, or didn't start within ``DRAFT_TRIAL_TIME_LIMIT_SECONDS``, has none of its rows
    matched."""
    deadline = time.monotonic() + DRAFT_TRIAL_TIME_LIMIT_SECONDS
    rows_matched = rows_tried = 0
    for reference, example in _drafting_examples():
        questions = QuestionSet(example.questions)
        [(key, question)] = questions.items()
        step = valid_step(method_step_data(reference, key, example.with_), BUILTINS.by_ref)
        rows_tried += len(example.rows)
        if time.monotonic() > deadline:
            continue
        try:
            drafted = draft_marking(
                QuestionSet({key: question.model_copy(update={"description": example.title})}),
                key,
                step.describe(),
                example.rows,
                model_endpoint=endpoint,
            )
        except model_service.ServiceError:
            raise
        except GradeFlowError:
            continue
        marked = mark(questions, MarkScheme([drafted.marking]), example.rows)
        rows_matched += sum(
            isclose(
                row.results[key].marks, row.sheet.expected[key].marks, abs_tol=ROUNDING_TOLERANCE
            )
            for row in marked
            if row.results[key].is_marked
        )
    return rows_matched, rows_tried


def _role_refusal(endpoint: model_service.ModelEndpoint, role: ModelRole, name: str) -> str | None:
    """Why one plain question shows ``name`` can't be the ``role`` model, or None."""
    system_prompt, prompt = "Answer with one word: Yes or No.", "Is 2 + 2 equal to 4?"
    if role == "fast":
        tokens = model_service.ask_token_logprobs(
            endpoint, name, system_prompt, prompt, max_tokens=4
        )
        return (
            None
            if model_service.yes_probabilities(tokens)
            else "Doesn't return token probabilities."
        )
    reply = model_service.ask(
        endpoint, name, system_prompt, prompt, reasoning_effort=None, max_tokens=4000
    )
    return None if "yes" in reply.lower() else "Didn't answer a simple question."
