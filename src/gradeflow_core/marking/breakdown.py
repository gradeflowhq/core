"""A result's breakdown: how each check of its marking marked the answer."""

from ..methods.outcome import Detail, Outcome, effect_of
from .results import CheckResult


def breakdown_of(outcome: Outcome, max_marks: float) -> CheckResult | None:
    """The breakdown rooted at ``outcome``; None when the marking is one check. Only the root's
    children say their marks, as deeper ones' depend on every check above."""
    if not outcome.children:
        return None
    children = tuple(_check_result(child, _marks(child, max_marks)) for child in outcome.children)
    return _check_result(outcome, None, children)


def _check_result(
    outcome: Outcome, marks: float | None, children: tuple[CheckResult, ...] | None = None
) -> CheckResult:
    effect = outcome.effect or effect_of(outcome)
    why = (Detail("Reason", outcome.feedback),) if effect == "unmarked" else ()
    if children is None:
        children = tuple(_check_result(child, None) for child in outcome.children)
    return CheckResult(
        title=_title(outcome),
        effect=effect,
        marks=marks,
        details=(*outcome.details, *why),
        children=children,
    )


def _marks(child: Outcome, max_marks: float) -> float | None:
    """The marks of the question ``child`` gave or took off, when its parent adds them up."""
    if child.contribution is None or child.effect in ("uncounted", "unmarked"):
        return None
    return child.contribution * max_marks


def _title(outcome: Outcome) -> str:
    label = ", ".join(outcome.labels)
    if not label:
        return outcome.title
    return f"**{label}:** {outcome.title}" if outcome.title else f"**{label}**"
