from collections.abc import Mapping
from typing import Any

from gradeflow_core import QuestionSet
from gradeflow_core.editing import MethodDraft
from gradeflow_core.questions.number import NumberQuestion

from ..methods.helpers import accepted_answer, mark_all, proportions

ACCEPTS_9_81 = accepted_answer("9.81", question="Q1")
ANSWERS = ["9.81", "-9.81", "9.8", "10"]
NUMBER_QUESTIONS = QuestionSet.model_validate({"Q1": {"type": "number", "max_marks": 2}})


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
