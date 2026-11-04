"""A result's breakdown: how each check of its marking marked the answer."""

from ..methods.outcome import Detail, Outcome, effect_of
from .results import CheckResult


def breakdown_of(outcome: Outcome, max_marks: float) -> CheckResult | None:
    """The breakdown rooted at ``outcome``; None when the marking is one check. Only the root's
    children say their marks, as deeper ones' depend on every check above."""
    if not outcome.children:
        return None
    # The result's own feedback says why it is unmarked, so the root gives no reason.
    return CheckResult(
        title=_title(outcome),
        effect=outcome.effect or effect_of(outcome),
        details=outcome.details,
        children=tuple(
            _check_result(child, _child_marks(child, max_marks)) for child in outcome.children
        ),
    )


def _check_result(outcome: Outcome, marks: float | None) -> CheckResult:
    effect = outcome.effect or effect_of(outcome)
    reason_details = (Detail("Reason", outcome.feedback),) if effect == "unmarked" else ()
    return CheckResult(
        title=_title(outcome),
        effect=effect,
        marks=marks,
        details=(*outcome.details, *reason_details),
        children=tuple(_check_result(child, None) for child in outcome.children),
    )


def _child_marks(child: Outcome, max_marks: float) -> float | None:
    """The marks of the question ``child`` gave, or its penalty, when its parent adds them up."""
    if child.contribution is None or child.effect in ("uncounted", "unmarked"):
        return None
    return child.contribution * max_marks


def _title(outcome: Outcome) -> str:
    label = ", ".join(outcome.labels)
    if not label:
        return outcome.title
    return f"**{label}:** {outcome.title}" if outcome.title else f"**{label}**"
