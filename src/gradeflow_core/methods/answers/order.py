from collections.abc import Callable, Iterator
from itertools import pairwise
from typing import Literal

from pydantic import Field, field_validator
from rapidfuzz.distance import LCSseq

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import markdown_code
from ...questions.base import Question
from ...questions.types import Answer, Value
from ...questions.values import ValuesQuestion
from ..check import Check
from ..example import examples_beside
from ..matching import read_or_none, reading_problems
from ..method import AnswerMethod
from ..outcome import CORRECT, Outcome
from ..position import Position
from ..settings import Settings, Text

MARKS_FOR_LABELS = {
    "exact": "Only the right order",
    "longest": "Items in order",
    "pairs": "Neighbours in order",
}


class OrderSettings(Settings):
    order: list[Text] = Field(
        min_length=2,
        title="Right order",
        description="First to last.",
        json_schema_extra={"uniqueItems": True, **hint(item="item")},
    )
    marks_for: Literal["exact", "longest", "pairs"] = Field(
        default="longest",
        title="Marks for",
        json_schema_extra=hint(labels=MARKS_FOR_LABELS),
    )

    @field_validator("order")
    @classmethod
    def _different(cls, order: list[str]) -> list[str]:
        if len(set(order)) < len(order):
            raise ValueError("Each item must be different.")
        return order


class Order(AnswerMethod[OrderSettings]):
    name = "gradeflow/order"
    version = 1
    label = "Order"
    summary = "Marks for putting items in the right order."
    guide = (
        "**Use it when:** Students put items in order, such as the stages of a process or events "
        "in time.\n\n"
        "**How it marks:** **Marks for** sets how the answer is compared with **Right order**. "
        "**Only the right order** gives full marks or none, and **Items in order** gives a share "
        "for the most items kept in order, so one item out of place costs one share. "
        "**Neighbours in order** gives a share for each pair side by side as in the right "
        "order.\n\n"
        "**Good to know:** The question needs one value for each item, all text or all "
        "numbers.\n\n"
        "**Related:** When each value has its own right answer, use Multiple Values."
    )
    question_types = frozenset({"values"})
    settings = OrderSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        partial = (
            ""
            if settings.marks_for == "exact"
            else f", with a share for {MARKS_FOR_LABELS[settings.marks_for].lower()}"
        )
        return f"Puts {', '.join(map(markdown_code, settings.order))} in order{partial}."

    def _mark_answer(self, settings: OrderSettings, answer: Answer, question: Question) -> Outcome:
        assert isinstance(question, ValuesQuestion)
        assert isinstance(answer, list)
        right = [read_or_none(_item_reading(question), item) for item in settings.order]
        expected = f"Expected {', '.join(settings.order)}."
        match settings.marks_for:
            case "exact":
                proportion = float(answer == right)
                said = expected
            case "longest":
                kept = LCSseq.similarity(answer, right)
                proportion = kept / len(right)
                said = f"{kept} of {len(right)} in order. {expected}"
            case "pairs":
                neighbours = len(set(pairwise(answer)) & set(pairwise(right)))
                proportion = neighbours / (len(right) - 1)
                said = f"{neighbours} of {len(right) - 1} pairs in order. {expected}"
        return Outcome(1.0, CORRECT) if proportion >= 1 else Outcome(proportion, said)

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        question = position.question_model
        if not isinstance(question, ValuesQuestion):
            return
        order = self._settings_of(check).order
        if question.value_count != len(order):
            yield Problem(
                f"List {question.value_count} items, one for each value.", field=("order",)
            )
        if len(set(question.value_types)) > 1:
            yield Problem(
                "The question's values must all be text or all numbers.", field=("order",)
            )
            return
        yield from reading_problems(order, _item_reading(question), "order")


def _item_reading(question: ValuesQuestion) -> Callable[[str], Value]:
    """How each item is read: as the first value is, the values being all of one type."""
    first = question.value_question(1)
    assert first is not None
    return first.read
