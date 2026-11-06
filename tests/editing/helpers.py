from collections.abc import Mapping, Sequence
from typing import Any, Literal

from gradeflow_core import BUILTINS, Methods, Question, QuestionSet
from gradeflow_core.editing import Decision, MarkingDraft, MethodDraft
from gradeflow_core.questions.number import NumberQuestion

from ..methods.helpers import accepted_answer, mark_all, proportions

ACCEPTS_9_81 = accepted_answer("9.81", question="Q1")
ANSWERS = ["9.81", "-9.81", "9.8", "10"]
NUMBER_QUESTIONS = QuestionSet.load({"Q1": {"type": "number", "max_marks": 2}})


def check_of(marking: Mapping[str, Any]) -> dict[str, Any]:
    """``marking`` as a check in another: without its question."""
    return {key: value for key, value in marking.items() if key != "question"}


def answer_proportions(marking: Mapping[str, Any]) -> list[float]:
    return proportions(mark_all(check_of(marking), ANSWERS, NumberQuestion()))


def number_range_draft() -> MethodDraft:
    draft = MethodDraft.new(
        NUMBER_QUESTIONS, name="range", version=1, label="Within a range", summary="In range."
    )
    draft.checks.add("with.checks.0", "gradeflow/number-range@1")
    draft.checks.update("with.checks.0", {"min_value": 10, "max_value": 20})
    return draft


def marked_by_example(
    marking: Mapping[str, Any],
    answers: Sequence[str],
    decision: Decision | Literal["cancel"],
    proportion: float = 0.5,
    *,
    question: Question | None = None,
    methods: Methods = BUILTINS,
) -> dict[str, Any]:
    draft = _draft(marking, question, methods)
    draft.mark_by_example(answers, decision, proportion)
    return draft.data


def decisions_of(
    marking: Mapping[str, Any], *, question: Question | None = None, methods: Methods = BUILTINS
) -> dict[str, Decision]:
    return _draft(marking, question, methods).answer_decisions()


def _draft(marking: Mapping[str, Any], question: Question | None, methods: Methods) -> MarkingDraft:
    """A draft of ``marking`` of Q1, or of no question to use the plain spelling of answers."""
    if question is None:
        return MarkingDraft(marking, QuestionSet(), None, methods=methods)
    return MarkingDraft(marking, QuestionSet({"Q1": question}), "Q1", methods=methods)
