"""Change the marks of particular answers without accumulating previous decisions."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from ..common.exceptions import EditingError
from ..methods.answers.accepted_answers import AcceptedAnswers
from ..methods.catalogue import BUILTINS
from ..methods.collection import Methods
from ..methods.logic.combination import Combination, combined_proportion
from ..methods.logic.full_marks import FullMarks
from ..methods.method import Method
from ..methods.outcome import FULL_MARKS_PROPORTION, clamped
from ..questions.base import Question

type Decision = Literal["full_marks", "part_marks", "not_allowed"]
DECISIONS: dict[Decision, str] = {
    "full_marks": "Full marks",
    "part_marks": "Part marks",
    "not_allowed": "Not allowed",
}
"""What marking by example may give answers, with how each reads; ``"cancel"`` takes it back."""
CANCEL = "cancel"

NOT_ALLOWED = "gradeflow/not-allowed"
"""The composed method that reverses a check."""


@dataclass(frozen=True)
class _ExampleMethods:
    """How the supplied methods write the checks that decide particular answers."""

    methods_by_reference: Mapping[str, Method[Any]]
    normalize: Callable[[str], str]
    accepted_answers_reference: str
    combination_reference: str
    not_allowed_reference: str
    full_marks_check: dict[str, Any]
    no_marks_check: dict[str, Any]
    """A check that gives no marks to any answer."""
    max_marks: float

    @classmethod
    def for_methods(cls, methods: Methods, question: Question | None) -> _ExampleMethods:
        latest_references = {method.name: method.ref for method in methods.latest()}
        missing = {
            AcceptedAnswers.name,
            Combination.name,
            NOT_ALLOWED,
            FullMarks.name,
        } - latest_references.keys()
        if missing:
            raise EditingError(f"There's no method {', '.join(sorted(missing))}.")
        return cls(
            methods_by_reference=methods.by_ref,
            normalize=question.normalize if question else str,
            accepted_answers_reference=latest_references[AcceptedAnswers.name],
            combination_reference=latest_references[Combination.name],
            not_allowed_reference=latest_references[NOT_ALLOWED],
            full_marks_check={"uses": latest_references[FullMarks.name]},
            no_marks_check={
                "uses": latest_references[AcceptedAnswers.name],
                "with": {"answers": []},
            },
            max_marks=question.max_marks if question else 1.0,
        )

    def uses_method(self, check: Mapping[str, Any], name: str) -> bool:
        method = self.methods_by_reference.get(str(check.get("uses")))
        return method is not None and method.name == name


def can_mark_by_example(question: Question) -> bool:
    return question.type in AcceptedAnswers.question_types


def _decision_for(proportion: float) -> Decision:
    return (
        "full_marks"
        if proportion >= FULL_MARKS_PROPORTION
        else "not_allowed"
        if proportion <= 0
        else "part_marks"
    )


def answer_decisions(
    marking: Mapping[str, Any],
    *,
    methods: Methods = BUILTINS,
    question: Question | None = None,
) -> dict[str, Decision]:
    """The answers given marks by example, as in accepted-answer, Not and Combination checks,
    with what each gets.

    Keys use the question's spelling. Other methods' matches are not decisions. A decision
    describes the explicit rule, independently of conditions and other methods' results.
    """
    proportions = _decided_proportions(
        marking, _ExampleMethods.for_methods(methods, question), marking.get("question")
    )
    return {answer: _decision_for(proportion) for answer, proportion in proportions.items()}


def _reads_same_question(check: Mapping[str, Any], key: str | None) -> bool:
    return (
        check.get("question") in (None, key)
        and check.get("value") is None
        and not check.get("read_only")
    )


def _decided_proportions(
    check: Mapping[str, Any], rules: _ExampleMethods, key: str | None
) -> dict[str, float]:
    if not _reads_same_question(check, key):
        return {}
    settings = check.get("with") or {}
    if rules.uses_method(check, AcceptedAnswers.name):
        return {rules.normalize(str(answer)): 1.0 for answer in settings.get("answers") or []}
    if rules.uses_method(check, NOT_ALLOWED):
        inner_check = settings.get("check") or {}
        return {
            answer: 1 - proportion
            for answer, proportion in _decided_proportions(inner_check, rules, key).items()
        }
    if not rules.uses_method(check, Combination.name):
        return {}
    inner_checks = [
        inner for inner in settings.get("checks") or [] if _reads_same_question(inner, key)
    ]
    combine = settings.get("combine", "all")
    weights = [inner.get("weight", 1) for inner in inner_checks]
    weighted_proportions: dict[str, list[tuple[float, float]]] = {}
    for inner in inner_checks:
        for answer, proportion in _decided_proportions(inner, rules, key).items():
            weighted_proportions.setdefault(answer, []).append((inner.get("weight", 1), proportion))
    return {
        answer: clamped(combined_proportion(combine, pairs, weights, rules.max_marks))
        for answer, pairs in weighted_proportions.items()
    }


def mark_by_example(
    marking: Mapping[str, Any],
    answers: Sequence[str],
    decision: Decision | Literal["cancel"],
    proportion: float = 0.5,
    *,
    methods: Methods = BUILTINS,
    question: Question | None = None,
) -> dict[str, Any]:
    """Gives these answers one of the ``DECISIONS``, ``proportion`` of the marks for
    ``"part_marks"``, or takes their decision back with ``"cancel"``. Other answers keep theirs.

    A decision overrides the existing marking. Cancelling leaves the remaining methods to mark
    the answers; an answer must be cancelled before it gets another decision, and giving one
    the decision it has changes nothing. Equivalent
    spellings are treated as one answer when ``question`` is supplied, which must be of a type
    that can be marked by example. New checks use the highest version of each method in
    ``methods``.
    """
    _check_decision(decision, question)
    rules = _ExampleMethods.for_methods(methods, question)
    chosen_answers = set(map(rules.normalize, answers))
    marking_copy = deepcopy(dict(marking))
    if not chosen_answers:
        return marking_copy
    if not 0 <= proportion <= 1:
        raise EditingError("The proportion of the marks must be between 0 and 1.")
    position_fields = {
        key: value for key, value in marking_copy.items() if key not in ("uses", "with")
    }
    check = {key: value for key, value in marking_copy.items() if key in ("uses", "with")}
    question_key = position_fields.get("question")
    decided_proportions = _decided_proportions(marking, rules, question_key)
    if decision == CANCEL:
        if chosen_answers.isdisjoint(decided_proportions):
            return marking_copy
        return {
            **position_fields,
            **_without_answers(check, chosen_answers, rules, cancel=True, key=question_key),
        }
    if not chosen_answers.isdisjoint(decided_proportions):
        if all(
            answer in decided_proportions and _decision_for(decided_proportions[answer]) == decision
            for answer in chosen_answers
        ):
            return marking_copy
        raise EditingError("These answers already have a decision.")
    check = _without_answers(check, chosen_answers, rules)
    accepted = {
        "uses": rules.accepted_answers_reference,
        "with": {"answers": list(dict.fromkeys(answers))},
    }
    if decision == "full_marks":
        check = _combine("any", [check, accepted], rules)
    else:
        if decision == "not_allowed" or not _excludes(check, chosen_answers, rules):
            not_accepted = {"uses": rules.not_allowed_reference, "with": {"check": accepted}}
            check = _combine("all", [check, not_accepted], rules)
        if decision == "part_marks":
            check = _combine("any", [check, {"weight": proportion, **accepted}], rules)
    return {**position_fields, **check}


def _check_decision(decision: str, question: Question | None) -> None:
    if decision not in (*DECISIONS, CANCEL):
        raise EditingError(f"There's no decision {decision}.")
    if question is not None and not can_mark_by_example(question):
        raise EditingError(f"A {question.label.lower()} question can't be marked by example.")


def _is_exact_list(check: Mapping[str, Any], rules: _ExampleMethods) -> bool:
    settings = check.get("with", {})
    return (
        rules.uses_method(check, AcceptedAnswers.name)
        and settings.get("match", "exact") == "exact"
        and not settings.get("tolerance")
    )


def _excludes(check: dict[str, Any], chosen: set[str], rules: _ExampleMethods) -> bool:
    """Whether ordinary exact checks already give these answers no marks."""
    if check == rules.no_marks_check:
        return True
    if set(check) & {"if", "read_only", "question", "value"}:
        return False
    settings = check.get("with", {})
    if rules.uses_method(check, AcceptedAnswers.name):
        return _is_exact_list(check, rules) and not chosen.intersection(
            map(rules.normalize, settings.get("answers", []))
        )
    if _is_boolean_combination(check, rules):
        inner_checks = settings.get("checks", [])
        match settings.get("combine", "all"):
            case "any":
                return all(_excludes(inner, chosen, rules) for inner in inner_checks)
            case "all":
                return any(_excludes(inner, chosen, rules) for inner in inner_checks)
    return False


def _without_answers(
    check: dict[str, Any],
    chosen: set[str],
    rules: _ExampleMethods,
    *,
    cancel: bool = False,
    key: str | None = None,
) -> dict[str, Any]:
    """Remove examples, simplifying only ordinary Boolean combinations.

    Replacement leaves fuzzy matches intact so an exact override affects only these answers.
    Cancellation also removes explicitly listed fuzzy examples, retaining check settings.
    """
    if cancel and not _reads_same_question(check, key):
        return check
    settings = check.get("with") or {}
    if rules.uses_method(check, AcceptedAnswers.name) and (cancel or _is_exact_list(check, rules)):
        remaining = [
            answer
            for answer in settings.get("answers") or []
            if rules.normalize(str(answer)) not in chosen
        ]
        if remaining:
            return {**check, "with": {**settings, "answers": remaining}}
        if set(check) <= {"uses", "with", "weight"}:
            return deepcopy(rules.no_marks_check)
        return {**check, **deepcopy(rules.no_marks_check)}
    if rules.uses_method(check, NOT_ALLOWED):
        inner = _without_answers(settings.get("check") or {}, chosen, rules, cancel=cancel, key=key)
        if _is_plain_check(check) and inner in (rules.no_marks_check, rules.full_marks_check):
            return deepcopy(
                rules.full_marks_check if inner == rules.no_marks_check else rules.no_marks_check
            )
        return {**check, "with": {**settings, "check": inner}}
    if rules.uses_method(check, Combination.name) and (
        cancel or _is_boolean_combination(check, rules)
    ):
        inner_checks = [
            _without_answers(inner, chosen, rules, cancel=cancel, key=key)
            for inner in settings.get("checks") or []
        ]
        if not inner_checks:
            return {**check, **deepcopy(rules.no_marks_check)}
        if (
            _is_plain_check(check)
            and _is_boolean_combination(check, rules)
            and set(settings) <= {"combine", "checks"}
        ):
            return _combine(settings.get("combine", "all"), inner_checks, rules)
        return {**check, "with": {**settings, "checks": inner_checks}}
    return check


def _is_plain_check(check: dict[str, Any]) -> bool:
    return set(check) <= {"uses", "with"}


def _is_boolean_combination(check: dict[str, Any], rules: _ExampleMethods) -> bool:
    settings = check.get("with", {})
    return (
        rules.uses_method(check, Combination.name)
        and settings.get("combine", "all") in ("all", "any")
        and not any(
            set(inner) & {"id", "if", "read_only", "question", "value"}
            for inner in settings.get("checks", [])
        )
    )


def _flattened_checks(
    check: dict[str, Any], mode: str, rules: _ExampleMethods
) -> list[dict[str, Any]]:
    settings = check.get("with", {})
    if (
        _is_plain_check(check)
        and _is_boolean_combination(check, rules)
        and settings.get("combine", "all") == mode
        and set(settings) <= {"combine", "checks"}
    ):
        return list(settings.get("checks", []))
    return [check]


def _combine(
    mode: str, inner_checks: list[dict[str, Any]], rules: _ExampleMethods
) -> dict[str, Any]:
    """Boolean identities keep repeated edits bounded; weights and other check settings stay put."""
    neutral_check = rules.full_marks_check if mode == "all" else rules.no_marks_check
    flattened: list[dict[str, Any]] = []
    for inner in inner_checks:
        for term in _flattened_checks(inner, mode, rules):
            if (
                mode == "any"
                and {key: value for key, value in term.items() if key != "weight"}
                == rules.no_marks_check
            ):
                continue
            if term != neutral_check and term not in flattened:
                flattened.append(term)
    if mode == "any":
        flattened = _merged_answer_lists(flattened, rules)
        exclusions, alternatives = _shared_exclusions(flattened, rules)
        if exclusions:
            alternatives_check = _combine("any", alternatives, rules)
            return _combine("all", [*exclusions, alternatives_check], rules)
    if not flattened:
        return deepcopy(neutral_check)
    if (
        len(flattened) == 1
        and flattened[0].get("weight", 1) == 1
        and not set(flattened[0]) & {"if", "read_only"}
    ):
        return flattened[0]
    return {"uses": rules.combination_reference, "with": {"combine": mode, "checks": flattened}}


def _shared_exclusions(
    inner_checks: list[dict[str, Any]], rules: _ExampleMethods
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Factor an exclusion out of alternatives only when none can award its rejected answers."""
    terms_by_check = [_flattened_checks(inner, "all", rules) for inner in inner_checks]
    exclusions: list[dict[str, Any]] = []
    for check_terms in terms_by_check:
        for check in check_terms:
            rejected = _rejected_answers(check, rules)
            if (
                rejected is not None
                and check not in exclusions
                and all(
                    check in terms or _excludes(inner, rejected, rules)
                    for inner, terms in zip(inner_checks, terms_by_check, strict=True)
                )
            ):
                exclusions.append(check)
    alternatives = [
        _combine("all", [check for check in check_terms if check not in exclusions], rules)
        if any(check in exclusions for check in check_terms)
        else inner
        for inner, check_terms in zip(inner_checks, terms_by_check, strict=True)
    ]
    return exclusions, alternatives


def _rejected_answers(check: dict[str, Any], rules: _ExampleMethods) -> set[str] | None:
    if not _is_plain_check(check) or not rules.uses_method(check, NOT_ALLOWED):
        return None
    inner = check.get("with", {}).get("check", {})
    if _is_plain_check(inner) and _is_exact_list(inner, rules):
        return set(map(rules.normalize, inner.get("with", {}).get("answers", [])))
    return None


def _merged_answer_lists(
    inner_checks: list[dict[str, Any]], rules: _ExampleMethods
) -> list[dict[str, Any]]:
    """Alternative answer lists with the matching settings and weight are one list."""

    def is_plain_list(check: dict[str, Any]) -> bool:
        return rules.uses_method(check, AcceptedAnswers.name) and set(check) <= {
            "uses",
            "with",
            "weight",
        }

    def settings(check: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in check.get("with", {}).items() if key != "answers"}

    joined: list[dict[str, Any]] = []
    for inner in inner_checks:
        matching = next(
            (
                other
                for other in joined
                if is_plain_list(inner)
                and is_plain_list(other)
                and other["uses"] == inner["uses"]
                and other.get("weight", 1) == inner.get("weight", 1)
                and settings(other) == settings(inner)
            ),
            None,
        )
        if matching is None:
            joined.append(inner)
        else:
            matching["with"]["answers"] = list(
                dict.fromkeys(matching["with"]["answers"] + inner["with"]["answers"])
            )
    return joined
