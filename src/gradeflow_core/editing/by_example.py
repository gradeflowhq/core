"""Change the marks of particular answers without accumulating previous decisions."""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from ..common.exceptions import EditingError
from ..methods.answers.accepted_answers import AcceptedAnswers, AcceptedAnswersSettings
from ..methods.collection import Methods
from ..methods.logic.full_marks import FullMarks
from ..questions.base import Question
from ..questions.types import QuestionId
from ..steps.combining import COMBINATIONS, MarksBy
from ..steps.method import Method
from ..steps.outcome import FULL_MARKS_PROPORTION
from ..steps.step import Step, load_step, settings_of
from ..steps.workflow import Workflow, WorkflowSettings

type Decision = Literal["full_marks", "partial_marks", "not_allowed"]
DECISIONS: dict[Decision, str] = {
    "full_marks": "Full marks",
    "partial_marks": "Partial marks",
    "not_allowed": "Not allowed",
}
"""What marking by example may give answers, with how each reads; ``"cancel"`` takes it back."""

NOT_ALLOWED_METHOD_NAME = "gradeflow/not-allowed"
"""The composed method that reverses a check."""

_METHOD_KEYS = frozenset({"uses", "with"})
"""What a check has of its method."""
_LIMIT_KEYS = frozenset({"if", "question", "input"})
"""What limits when a check runs or which answer it reads."""


@dataclass(frozen=True)
class _ExampleMethods:
    """How the supplied methods write the checks that decide particular answers."""

    methods_by_reference: Mapping[str, Method]
    normalize: Callable[[str], str]
    accepted_answers_reference: str
    workflow_reference: str
    not_allowed_reference: str
    full_marks_check: dict[str, Any]
    no_marks_check: dict[str, Any]
    """A check that gives no marks to any answer."""
    max_marks: float
    question_id: QuestionId | None
    """The marking's question."""

    @classmethod
    def for_methods(
        cls, methods: Methods, question: Question | None, question_id: QuestionId | None
    ) -> _ExampleMethods:
        latest_references = {method.name: method.ref for method in methods.latest()}
        missing_method_names = {
            AcceptedAnswers.name,
            Workflow.name,
            NOT_ALLOWED_METHOD_NAME,
            FullMarks.name,
        } - latest_references.keys()
        if missing_method_names:
            raise EditingError(f"There's no method {', '.join(sorted(missing_method_names))}.")
        return cls(
            methods_by_reference=methods.by_ref,
            normalize=question.normalize if question else str,
            accepted_answers_reference=latest_references[AcceptedAnswers.name],
            workflow_reference=latest_references[Workflow.name],
            not_allowed_reference=latest_references[NOT_ALLOWED_METHOD_NAME],
            full_marks_check={"uses": latest_references[FullMarks.name]},
            no_marks_check={
                "uses": latest_references[AcceptedAnswers.name],
                "with": {"answers": []},
            },
            max_marks=question.max_marks if question else 1.0,
            question_id=question_id,
        )

    def uses_method(self, check: Mapping[str, Any], name: str) -> bool:
        method = self.methods_by_reference.get(str(check.get("uses")))
        return method is not None and method.name == name


def _steps(check: Mapping[str, Any]) -> list[Any]:
    steps = settings_of(check).get("steps")
    return steps if isinstance(steps, list) else []


def _answers(check: Mapping[str, Any]) -> list[Any]:
    return settings_of(check).get("answers") or []


def _marks_by(check: Mapping[str, Any]) -> object:
    return settings_of(check).get("marks_by", WorkflowSettings.model_fields["marks_by"].default)


def _weight(check: Mapping[str, Any]) -> object:
    return check.get("weight", Step.model_fields["weight"].default)


def can_mark_by_example(question: Question) -> bool:
    return question.type in AcceptedAnswers.question_types


def _decision_for(proportion: float) -> Decision:
    return (
        "full_marks"
        if proportion >= FULL_MARKS_PROPORTION
        else "not_allowed"
        if proportion <= 0
        else "partial_marks"
    )


def answer_decisions(
    marking: Mapping[str, Any], methods: Methods, question: Question | None
) -> dict[str, Decision]:
    proportions = _decided_proportions(
        marking, _ExampleMethods.for_methods(methods, question, marking.get("question"))
    )
    return {answer: _decision_for(proportion) for answer, proportion in proportions.items()}


def _decided_proportions(
    marking: Mapping[str, Any], example_methods: _ExampleMethods
) -> dict[str, float]:
    """The proportion of the marks ``marking`` gives each answer it decides; none until it names
    a method."""
    if "uses" not in marking:
        return {}
    step = load_step(marking, example_methods.methods_by_reference)
    return _proportions_decided_by(step, example_methods)


def _proportions_decided_by(
    step: Step | None, example_methods: _ExampleMethods
) -> dict[str, float]:
    """The proportion of the marks ``step`` gives each answer it decides. A step that isn't
    there, or whose settings aren't valid, as while it is being written, decides none."""
    if step is None or not _marks_same_question(step, example_methods):
        return {}
    method = step.resolution.method
    if method is not None and method.name == NOT_ALLOWED_METHOD_NAME:
        reversed_check = step.resolution.inner_steps.get("/with/check")
        return {
            answer: 1 - proportion
            for answer, proportion in _proportions_decided_by(
                reversed_check, example_methods
            ).items()
        }
    match step.settings:
        case AcceptedAnswersSettings(answers=answers):
            return {example_methods.normalize(answer): 1.0 for answer in answers}
        case WorkflowSettings(marks_by=marks_by, steps=steps):
            checks = [check for check in steps if _marks_same_question(check, example_methods)]
            combination = COMBINATIONS[marks_by]
            weighted_proportions: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
            for check in checks:
                decided_proportions = _proportions_decided_by(check, example_methods)
                for answer, proportion in decided_proportions.items():
                    weighted_proportions[answer].append((check.weight, proportion))
            denominator = combination.denominator(
                [check.weight for check in checks], example_methods.max_marks
            )
            return {
                answer: combination.proportion(weight_proportion_pairs, denominator)
                for answer, weight_proportion_pairs in weighted_proportions.items()
            }
    return {}


def _marks_same_question(step: Step, example_methods: _ExampleMethods) -> bool:
    """Whether ``step`` gives marks for the marking's own answer."""
    return (
        step.question in (None, example_methods.question_id)
        and step.input is None
        and step.weight != 0
    )


def mark_by_example(
    marking: Mapping[str, Any],
    answers: Sequence[str],
    decision: Decision | Literal["cancel"],
    proportion: float,
    methods: Methods,
    question: Question | None,
) -> dict[str, Any] | None:
    """``marking`` with ``answers`` given ``decision``; None when that changes nothing."""
    _validate_decision(decision, question)
    if not 0 <= proportion <= 1:
        raise EditingError("The proportion of the marks must be between 0 and 1.")
    example_methods = _ExampleMethods.for_methods(methods, question, marking.get("question"))
    chosen_answers = set(map(example_methods.normalize, answers))
    if not chosen_answers:
        return None
    position_fields = {key: value for key, value in marking.items() if key not in _METHOD_KEYS}
    check = {key: value for key, value in marking.items() if key in _METHOD_KEYS}
    decided_proportions = _decided_proportions(marking, example_methods)
    if decision == "cancel":
        if chosen_answers.isdisjoint(decided_proportions):
            return None
        return {
            **position_fields,
            **_without_answers(check, chosen_answers, example_methods, cancel=True),
        }
    if not chosen_answers.isdisjoint(decided_proportions):
        if all(
            answer in decided_proportions and _decision_for(decided_proportions[answer]) == decision
            for answer in chosen_answers
        ):
            return None
        raise EditingError("These answers already have a decision.")
    check = _without_answers(check, chosen_answers, example_methods)
    accepted_answers_check = {
        "uses": example_methods.accepted_answers_reference,
        "with": {"answers": list(dict.fromkeys(answers))},
    }
    if decision == "full_marks":
        check = _combined_check("any_correct", [check, accepted_answers_check], example_methods)
    else:
        if decision == "not_allowed" or not _excludes(check, chosen_answers, example_methods):
            not_allowed_check = {
                "uses": example_methods.not_allowed_reference,
                "with": {"check": accepted_answers_check},
            }
            check = _combined_check("all_correct", [check, not_allowed_check], example_methods)
        if decision == "partial_marks":
            check = _combined_check(
                "any_correct",
                [check, {"weight": proportion, **accepted_answers_check}],
                example_methods,
            )
    return {**position_fields, **check}


def _validate_decision(decision: str, question: Question | None) -> None:
    if decision not in (*DECISIONS, "cancel"):
        raise EditingError(f"There's no decision {decision}.")
    if question is not None and not can_mark_by_example(question):
        raise EditingError(f"A {question.label.lower()} question can't be marked by example.")


def _is_exact_answer_list(check: Mapping[str, Any], example_methods: _ExampleMethods) -> bool:
    settings = settings_of(check)
    return (
        example_methods.uses_method(check, AcceptedAnswers.name)
        and settings.get("match", "exact") == "exact"
        and not settings.get("tolerance")
    )


def _excludes(
    check: dict[str, Any], chosen_answers: set[str], example_methods: _ExampleMethods
) -> bool:
    """Whether ordinary exact checks already give these answers no marks."""
    if check == example_methods.no_marks_check:
        return True
    if set(check) & _LIMIT_KEYS:
        return False
    if example_methods.uses_method(check, AcceptedAnswers.name):
        return _is_exact_answer_list(check, example_methods) and not chosen_answers.intersection(
            map(example_methods.normalize, _answers(check))
        )
    if _is_all_or_any_workflow(check, example_methods):
        step_checks = _steps(check)
        match _marks_by(check):
            case "any_correct":
                return all(
                    _excludes(step_check, chosen_answers, example_methods)
                    for step_check in step_checks
                )
            case "all_correct":
                return any(
                    _excludes(step_check, chosen_answers, example_methods)
                    for step_check in step_checks
                )
    return False


def _without_answers(
    check: dict[str, Any],
    chosen_answers: set[str],
    example_methods: _ExampleMethods,
    *,
    cancel: bool = False,
) -> dict[str, Any]:
    """Remove examples, simplifying only all-or-any workflows.

    Replacement leaves fuzzy matches intact so an exact override affects only these answers.
    Cancellation also removes explicitly listed fuzzy examples, retaining check settings.
    """
    if cancel and not (
        check.get("question") in (None, example_methods.question_id)
        and check.get("input") is None
        and _weight(check) != 0
    ):
        return check
    settings = settings_of(check)
    if example_methods.uses_method(check, AcceptedAnswers.name) and (
        cancel or _is_exact_answer_list(check, example_methods)
    ):
        remaining_answers = [
            answer
            for answer in _answers(check)
            if example_methods.normalize(str(answer)) not in chosen_answers
        ]
        if remaining_answers:
            return {**check, "with": {**settings, "answers": remaining_answers}}
        return {**check, **deepcopy(example_methods.no_marks_check)}
    if example_methods.uses_method(check, NOT_ALLOWED_METHOD_NAME):
        reversed_check = _without_answers(
            settings.get("check") or {}, chosen_answers, example_methods, cancel=cancel
        )
        if _is_plain_check(check) and reversed_check in (
            example_methods.no_marks_check,
            example_methods.full_marks_check,
        ):
            return deepcopy(
                example_methods.full_marks_check
                if reversed_check == example_methods.no_marks_check
                else example_methods.no_marks_check
            )
        return {**check, "with": {**settings, "check": reversed_check}}
    if example_methods.uses_method(check, Workflow.name) and (
        cancel or _is_all_or_any_workflow(check, example_methods)
    ):
        step_checks = [
            _without_answers(step_check, chosen_answers, example_methods, cancel=cancel)
            for step_check in _steps(check)
        ]
        if not step_checks:
            return {**check, **deepcopy(example_methods.no_marks_check)}
        if _is_plain_all_or_any_workflow(check, example_methods):
            marks_by: MarksBy = (
                "any_correct" if _marks_by(check) == "any_correct" else "all_correct"
            )
            return _combined_check(marks_by, step_checks, example_methods)
        return {**check, "with": {**settings, "steps": step_checks}}
    return check


def _is_plain_check(check: dict[str, Any]) -> bool:
    return set(check) <= _METHOD_KEYS


def _is_all_or_any_workflow(check: dict[str, Any], example_methods: _ExampleMethods) -> bool:
    steps = _steps(check)
    return (
        example_methods.uses_method(check, Workflow.name)
        and _marks_by(check) in ("all_correct", "any_correct")
        and not any(set(step_check) & {"id", *_LIMIT_KEYS} for step_check in steps)
        and all(_weight(step_check) != 0 for step_check in steps)
    )


def _is_plain_all_or_any_workflow(check: dict[str, Any], example_methods: _ExampleMethods) -> bool:
    return (
        _is_plain_check(check)
        and _is_all_or_any_workflow(check, example_methods)
        and set(settings_of(check)) <= {"marks_by", "steps"}
    )


def _flattened_checks(
    check: dict[str, Any], marks_by: MarksBy, example_methods: _ExampleMethods
) -> list[dict[str, Any]]:
    if _is_plain_all_or_any_workflow(check, example_methods) and _marks_by(check) == marks_by:
        return _steps(check)
    return [check]


def _combined_check(
    marks_by: MarksBy, checks: list[dict[str, Any]], example_methods: _ExampleMethods
) -> dict[str, Any]:
    """Boolean identities keep repeated edits bounded; weights and other check settings stay put."""
    neutral_check = (
        example_methods.full_marks_check
        if marks_by == "all_correct"
        else example_methods.no_marks_check
    )
    flattened_checks: list[dict[str, Any]] = []
    for check in checks:
        for flattened_check in _flattened_checks(check, marks_by, example_methods):
            if (
                marks_by == "any_correct"
                and {key: value for key, value in flattened_check.items() if key != "weight"}
                == example_methods.no_marks_check
            ):
                continue
            if flattened_check != neutral_check and flattened_check not in flattened_checks:
                flattened_checks.append(flattened_check)
    if marks_by == "any_correct":
        flattened_checks = _merged_answer_lists(flattened_checks, example_methods)
        exclusions, alternatives = _shared_exclusions(flattened_checks, example_methods)
        if exclusions:
            alternatives_check = _combined_check("any_correct", alternatives, example_methods)
            return _combined_check(
                "all_correct", [*exclusions, alternatives_check], example_methods
            )
    if not flattened_checks:
        return deepcopy(neutral_check)
    if (
        len(flattened_checks) == 1
        and _weight(flattened_checks[0]) == 1
        and "if" not in flattened_checks[0]
    ):
        return flattened_checks[0]
    return {
        "uses": example_methods.workflow_reference,
        "with": {"marks_by": marks_by, "steps": flattened_checks},
    }


def _shared_exclusions(
    checks: list[dict[str, Any]], example_methods: _ExampleMethods
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Factor an exclusion out of alternatives only when none can award its rejected answers."""
    conjuncts_per_check = [
        _flattened_checks(check, "all_correct", example_methods) for check in checks
    ]
    exclusions: list[dict[str, Any]] = []
    for conjuncts in conjuncts_per_check:
        for conjunct in conjuncts:
            rejected_answers = _rejected_answers(conjunct, example_methods)
            if (
                rejected_answers is not None
                and conjunct not in exclusions
                and all(
                    conjunct in conjuncts_of_check
                    or _excludes(check, rejected_answers, example_methods)
                    for check, conjuncts_of_check in zip(checks, conjuncts_per_check, strict=True)
                )
            ):
                exclusions.append(conjunct)
    alternatives = [
        _combined_check(
            "all_correct",
            [conjunct for conjunct in conjuncts if conjunct not in exclusions],
            example_methods,
        )
        if any(conjunct in exclusions for conjunct in conjuncts)
        else check
        for check, conjuncts in zip(checks, conjuncts_per_check, strict=True)
    ]
    return exclusions, alternatives


def _rejected_answers(check: dict[str, Any], example_methods: _ExampleMethods) -> set[str] | None:
    if not _is_plain_check(check) or not example_methods.uses_method(
        check, NOT_ALLOWED_METHOD_NAME
    ):
        return None
    reversed_check = settings_of(check).get("check") or {}
    if _is_plain_check(reversed_check) and _is_exact_answer_list(reversed_check, example_methods):
        return set(map(example_methods.normalize, _answers(reversed_check)))
    return None


def _merged_answer_lists(
    checks: list[dict[str, Any]], example_methods: _ExampleMethods
) -> list[dict[str, Any]]:
    """Alternative answer lists with the matching settings and weight are one list."""

    def is_plain_list(check: dict[str, Any]) -> bool:
        return (
            example_methods.uses_method(check, AcceptedAnswers.name)
            and set(check) - {"weight"} <= _METHOD_KEYS
        )

    def settings_besides_answers(check: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in settings_of(check).items() if key != "answers"}

    merged_checks: list[dict[str, Any]] = []
    for check in checks:
        index = next(
            (
                index
                for index, merged_check in enumerate(merged_checks)
                if is_plain_list(check)
                and is_plain_list(merged_check)
                and merged_check["uses"] == check["uses"]
                and _weight(merged_check) == _weight(check)
                and settings_besides_answers(merged_check) == settings_besides_answers(check)
            ),
            None,
        )
        if index is None:
            merged_checks.append(check)
        else:
            answers = list(dict.fromkeys([*_answers(merged_checks[index]), *_answers(check)]))
            merged_checks[index] = {
                **merged_checks[index],
                "with": {**settings_of(merged_checks[index]), "answers": answers},
            }
    return merged_checks
