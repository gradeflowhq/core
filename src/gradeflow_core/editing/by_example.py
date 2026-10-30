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
from ..methods.outcome import FULL_MARKS, clamped
from ..questions.base import Question

type Decision = Literal["full", "part", "not_allowed"]
DECISIONS: dict[Decision, str] = {
    "full": "Full marks",
    "part": "Part marks",
    "not_allowed": "Not allowed",
}
"""What marking by example may give answers, with how each reads; ``"cancel"`` takes it back."""
CANCEL = "cancel"

NOT = "gradeflow/not-allowed"
"""The composed method that reverses a check."""


@dataclass(frozen=True)
class _Rules:
    """How the supplied methods write the checks that decide particular answers."""

    lookup: Mapping[str, Method[Any]]
    normalize: Callable[[str], str]
    accepted: str
    combination: str
    not_: str
    full: dict[str, Any]
    zero: dict[str, Any]
    """A check that gives no marks to any answer."""
    max_marks: float

    @classmethod
    def of(cls, methods: Methods, question: Question | None) -> _Rules:
        newest = {method.name: method.ref for method in methods.latest()}
        missing = {AcceptedAnswers.name, Combination.name, NOT, FullMarks.name} - newest.keys()
        if missing:
            raise EditingError(f"There's no method {', '.join(sorted(missing))}.")
        full = {"uses": newest[FullMarks.name]}
        return cls(
            lookup=methods.by_ref,
            normalize=question.normalize if question else str,
            accepted=newest[AcceptedAnswers.name],
            combination=newest[Combination.name],
            not_=newest[NOT],
            full=full,
            zero={"uses": newest[AcceptedAnswers.name], "with": {"answers": []}},
            max_marks=question.max_marks if question else 1.0,
        )

    def uses(self, check: Mapping[str, Any], name: str) -> bool:
        method = self.lookup.get(str(check.get("uses")))
        return method is not None and method.name == name


def can_mark_by_example(question: Question) -> bool:
    return question.type in AcceptedAnswers.question_types


def _decision(proportion: float) -> Decision:
    return "full" if proportion >= FULL_MARKS else "not_allowed" if proportion <= 0 else "part"


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
    proportions = _decisions(marking, _Rules.of(methods, question), marking.get("question"))
    return {answer: _decision(proportion) for answer, proportion in proportions.items()}


def _same_question(check: Mapping[str, Any], key: str | None) -> bool:
    return (
        check.get("question") in (None, key)
        and check.get("value") is None
        and not check.get("read_only")
    )


def _decisions(check: Mapping[str, Any], rules: _Rules, key: str | None) -> dict[str, float]:
    if not _same_question(check, key):
        return {}
    values = check.get("with") or {}
    if rules.uses(check, AcceptedAnswers.name):
        return {rules.normalize(str(answer)): 1.0 for answer in values.get("answers") or []}
    if rules.uses(check, NOT):
        return {
            answer: 1 - proportion
            for answer, proportion in _decisions(values.get("check") or {}, rules, key).items()
        }
    if not rules.uses(check, Combination.name):
        return {}
    children = [child for child in values.get("checks") or [] if _same_question(child, key)]
    mode = values.get("combine", "all")
    weights = [child.get("weight", 1) for child in children]
    weighted: dict[str, list[tuple[float, float]]] = {}
    for child in children:
        for answer, proportion in _decisions(child, rules, key).items():
            weighted.setdefault(answer, []).append((child.get("weight", 1), proportion))
    return {
        answer: clamped(combined_proportion(mode, parts, weights, rules.max_marks))
        for answer, parts in weighted.items()
    }


def mark_by_example(
    marking: Mapping[str, Any],
    answers: Sequence[str],
    given: Decision | Literal["cancel"],
    proportion: float = 0.5,
    *,
    methods: Methods = BUILTINS,
    question: Question | None = None,
) -> dict[str, Any]:
    """Gives these answers one of the ``DECISIONS``, ``proportion`` of the marks for ``"part"``, or
    takes their decision back with ``"cancel"``. Other answers keep theirs.

    A decision overrides the existing marking. Cancelling leaves the remaining methods to mark
    the answers; an answer must be cancelled before it gets another decision, and giving one
    the decision it has changes nothing. Equivalent
    spellings are treated as one answer when ``question`` is supplied, which must be of a type
    that can be marked by example. New checks use the highest version of each method in
    ``methods``.
    """
    _refuse(given, question)
    rules = _Rules.of(methods, question)
    chosen = set(map(rules.normalize, answers))
    unchanged = deepcopy(dict(marking))
    if not chosen:
        return unchanged
    if not 0 <= proportion <= 1:
        raise EditingError("The proportion of the marks must be between 0 and 1.")
    outer = {key: value for key, value in unchanged.items() if key not in ("uses", "with")}
    check = {key: value for key, value in unchanged.items() if key in ("uses", "with")}
    decided = _decisions(marking, rules, outer.get("question"))
    if given == CANCEL:
        if chosen.isdisjoint(decided):
            return unchanged
        return {**outer, **_without(check, chosen, rules, cancel=True, key=outer.get("question"))}
    if not chosen.isdisjoint(decided):
        if all(answer in decided and _decision(decided[answer]) == given for answer in chosen):
            return unchanged
        raise EditingError("These answers already have a decision.")
    check = _without(check, chosen, rules)
    accepted = {"uses": rules.accepted, "with": {"answers": list(dict.fromkeys(answers))}}
    if given == "full":
        check = _combine("any", [check, accepted], rules)
    else:
        if given == "not_allowed" or not _excludes(check, chosen, rules):
            not_accepted = {"uses": rules.not_, "with": {"check": accepted}}
            check = _combine("all", [check, not_accepted], rules)
        if given == "part":
            check = _combine("any", [check, {"weight": proportion, **accepted}], rules)
    return {**outer, **check}


def _refuse(given: str, question: Question | None) -> None:
    if given not in (*DECISIONS, CANCEL):
        raise EditingError(f"There's no decision {given}.")
    if question is not None and not can_mark_by_example(question):
        raise EditingError(f"A {question.label.lower()} question can't be marked by example.")


def _exact(check: Mapping[str, Any], rules: _Rules) -> bool:
    values = check.get("with", {})
    return (
        rules.uses(check, AcceptedAnswers.name)
        and values.get("match", "exact") == "exact"
        and not values.get("tolerance")
    )


def _excludes(check: dict[str, Any], chosen: set[str], rules: _Rules) -> bool:
    """Whether ordinary exact checks already give these answers no marks."""
    if check == rules.zero:
        return True
    if set(check) & {"if", "read_only", "question", "value"}:
        return False
    values = check.get("with", {})
    if rules.uses(check, AcceptedAnswers.name):
        return _exact(check, rules) and not chosen.intersection(
            map(rules.normalize, values.get("answers", []))
        )
    if _boolean(check, rules):
        children = values.get("checks", [])
        match values.get("combine", "all"):
            case "any":
                return all(_excludes(child, chosen, rules) for child in children)
            case "all":
                return any(_excludes(child, chosen, rules) for child in children)
    return False


def _without(
    check: dict[str, Any],
    chosen: set[str],
    rules: _Rules,
    *,
    cancel: bool = False,
    key: str | None = None,
) -> dict[str, Any]:
    """Remove examples, simplifying only ordinary Boolean combinations.

    Replacement leaves fuzzy matches intact so an exact override affects only these answers.
    Cancellation also removes explicitly listed fuzzy examples, retaining check settings.
    """
    if cancel and not _same_question(check, key):
        return check
    values = check.get("with") or {}
    if rules.uses(check, AcceptedAnswers.name) and (cancel or _exact(check, rules)):
        remaining = [
            answer
            for answer in values.get("answers") or []
            if rules.normalize(str(answer)) not in chosen
        ]
        if remaining:
            return {**check, "with": {**values, "answers": remaining}}
        if set(check) <= {"uses", "with", "weight"}:
            return deepcopy(rules.zero)
        return {**check, **deepcopy(rules.zero)}
    if rules.uses(check, NOT):
        child = _without(values.get("check") or {}, chosen, rules, cancel=cancel, key=key)
        if _bare(check) and child in (rules.zero, rules.full):
            return deepcopy(rules.full if child == rules.zero else rules.zero)
        return {**check, "with": {**values, "check": child}}
    if rules.uses(check, Combination.name) and (cancel or _boolean(check, rules)):
        children = [
            _without(child, chosen, rules, cancel=cancel, key=key)
            for child in values.get("checks") or []
        ]
        if not children:
            return {**check, **deepcopy(rules.zero)}
        if _bare(check) and _boolean(check, rules) and set(values) <= {"combine", "checks"}:
            return _combine(values.get("combine", "all"), children, rules)
        return {**check, "with": {**values, "checks": children}}
    return check


def _bare(check: dict[str, Any]) -> bool:
    return set(check) <= {"uses", "with"}


def _boolean(check: dict[str, Any], rules: _Rules) -> bool:
    values = check.get("with", {})
    return (
        rules.uses(check, Combination.name)
        and values.get("combine", "all") in ("all", "any")
        and not any(
            set(child) & {"id", "if", "read_only", "question", "value"}
            for child in values.get("checks", [])
        )
    )


def _terms(check: dict[str, Any], mode: str, rules: _Rules) -> list[dict[str, Any]]:
    values = check.get("with", {})
    if (
        _bare(check)
        and _boolean(check, rules)
        and values.get("combine", "all") == mode
        and set(values) <= {"combine", "checks"}
    ):
        return list(values.get("checks", []))
    return [check]


def _combine(mode: str, children: list[dict[str, Any]], rules: _Rules) -> dict[str, Any]:
    """Boolean identities keep repeated edits bounded; weights and other check settings stay put."""
    identity = rules.full if mode == "all" else rules.zero
    flattened: list[dict[str, Any]] = []
    for child in children:
        for term in _terms(child, mode, rules):
            if (
                mode == "any"
                and {key: value for key, value in term.items() if key != "weight"} == rules.zero
            ):
                continue
            if term != identity and term not in flattened:
                flattened.append(term)
    if mode == "any":
        flattened = _joined(flattened, rules)
        exclusions, alternatives = _shared_exclusions(flattened, rules)
        if exclusions:
            awards = _combine("any", alternatives, rules)
            return _combine("all", [*exclusions, awards], rules)
    if not flattened:
        return deepcopy(identity)
    if (
        len(flattened) == 1
        and flattened[0].get("weight", 1) == 1
        and not set(flattened[0]) & {"if", "read_only"}
    ):
        return flattened[0]
    return {"uses": rules.combination, "with": {"combine": mode, "checks": flattened}}


def _shared_exclusions(
    children: list[dict[str, Any]], rules: _Rules
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Factor an exclusion out of alternatives only when none can award its rejected answers."""
    groups = [_terms(child, "all", rules) for child in children]
    exclusions: list[dict[str, Any]] = []
    for group in groups:
        for check in group:
            rejected = _rejected(check, rules)
            if (
                rejected is not None
                and check not in exclusions
                and all(
                    check in terms or _excludes(child, rejected, rules)
                    for child, terms in zip(children, groups, strict=True)
                )
            ):
                exclusions.append(check)
    alternatives = [
        _combine("all", [check for check in group if check not in exclusions], rules)
        if any(check in exclusions for check in group)
        else child
        for child, group in zip(children, groups, strict=True)
    ]
    return exclusions, alternatives


def _rejected(check: dict[str, Any], rules: _Rules) -> set[str] | None:
    if not _bare(check) or not rules.uses(check, NOT):
        return None
    child = check.get("with", {}).get("check", {})
    if _bare(child) and _exact(child, rules):
        return set(map(rules.normalize, child.get("with", {}).get("answers", [])))
    return None


def _joined(children: list[dict[str, Any]], rules: _Rules) -> list[dict[str, Any]]:
    """Alternative answer lists with the same settings and weight are one list."""

    def plain_list(check: dict[str, Any]) -> bool:
        return rules.uses(check, AcceptedAnswers.name) and set(check) <= {"uses", "with", "weight"}

    def settings(check: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in check.get("with", {}).items() if key != "answers"}

    joined: list[dict[str, Any]] = []
    for child in children:
        same = next(
            (
                other
                for other in joined
                if plain_list(child)
                and plain_list(other)
                and other["uses"] == child["uses"]
                and other.get("weight", 1) == child.get("weight", 1)
                and settings(other) == settings(child)
            ),
            None,
        )
        if same is None:
            joined.append(child)
        else:
            same["with"]["answers"] = list(
                dict.fromkeys(same["with"]["answers"] + child["with"]["answers"])
            )
    return joined
