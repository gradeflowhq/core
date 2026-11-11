from collections.abc import Callable, Iterator
from itertools import pairwise
from typing import Literal

from pydantic import Field
from rapidfuzz.distance import LCSseq

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import code_span, reads_better_listed
from ...questions.base import Question
from ...questions.types import Answer, Value
from ...questions.values import ValuesQuestion
from ...steps.example import examples_beside
from ...steps.method import AnswerMethod
from ...steps.outcome import CORRECT_FEEDBACK, Outcome
from ...steps.position import Position
from ...steps.settings import NonBlankText, Settings
from ..matching import duplicate_problems, unreadable_problems

MARKS_FOR_LABELS = {
    "exact": "Only the correct order",
    "items_in_order": "Items in order",
    "pairs_in_order": "Pairs in order",
}


class OrderSettings(Settings):
    order: list[NonBlankText] = Field(
        min_length=2,
        title="Correct order",
        description="First to last.",
        json_schema_extra={"uniqueItems": True, **hint(item_label="item")},
    )
    marks_for: Literal["exact", "items_in_order", "pairs_in_order"] = Field(
        default="items_in_order",
        title="Marks for",
        json_schema_extra=hint(labels=MARKS_FOR_LABELS),
    )


class Order(AnswerMethod[OrderSettings]):
    name = "gradeflow/order"
    version = 1
    label = "Order"
    summary = "Marks for putting items in the correct order."
    guide = (
        "**Use it when:** Students put items in order, such as the stages of a process or events "
        "in time.\n\n"
        "**How it marks:** **Marks for** sets how the answer is compared with **Correct order**. "
        "**Only the correct order** gives full marks or none, and **Items in order** gives a share "
        "for the most items kept in order, so one item out of place costs one share. "
        "**Pairs in order** gives a share for each pair side by side as in the correct "
        "order.\n\n"
        "**Good to know:** The question needs one value for each item, all text or all "
        "numbers.\n\n"
        "**Related:** When each value has its own correct answer, use Multiple Values."
    )
    question_types = frozenset({"values"})
    settings = OrderSettings
    examples = examples_beside(__file__)

    def describe(self, settings: OrderSettings) -> str:
        partial_marks_note = (
            ""
            if settings.marks_for == "exact"
            else f", with partial marks for {MARKS_FOR_LABELS[settings.marks_for].lower()}"
        )
        if reads_better_listed(settings.order):
            item_lines = "\n".join(
                f"{place}. {code_span(item)}" for place, item in enumerate(settings.order, 1)
            )
            return f"Puts these in order{partial_marks_note}:\n{item_lines}"
        return f"Puts {', '.join(map(code_span, settings.order))} in order{partial_marks_note}."

    def mark_answer(self, settings: OrderSettings, answer: Answer, question: Question) -> Outcome:
        assert isinstance(question, ValuesQuestion)
        assert isinstance(answer, list)
        correct_order = list(map(_item_reader(question), settings.order))
        expected_sentence = f"Expected {', '.join(settings.order)}."
        match settings.marks_for:
            case "exact":
                proportion = float(answer == correct_order)
                feedback = expected_sentence
            case "items_in_order":
                items_in_order_count = LCSseq.similarity(answer, correct_order)
                proportion = items_in_order_count / len(correct_order)
                feedback = (
                    f"{items_in_order_count} of {len(correct_order)} in order. {expected_sentence}"
                )
            case "pairs_in_order":
                pairs_in_order_count = len(set(pairwise(answer)) & set(pairwise(correct_order)))
                proportion = pairs_in_order_count / (len(correct_order) - 1)
                feedback = (
                    f"{pairs_in_order_count} of {len(correct_order) - 1} pairs in order. "
                    f"{expected_sentence}"
                )
        return Outcome(1.0, CORRECT_FEEDBACK) if proportion >= 1 else Outcome(proportion, feedback)

    def problems(self, settings: OrderSettings, position: Position) -> Iterator[Problem]:
        question = position.question_model
        if not isinstance(question, ValuesQuestion):
            return
        order = settings.order
        if question.value_count != len(order):
            yield Problem(f"List {question.value_count} items, one for each value.", field="/order")
        if len(set(question.value_types)) > 1:
            yield Problem("The question's values must all be text or all numbers.", field="/order")
            return
        read = _item_reader(question)
        yield from unreadable_problems(order, read, "order")
        yield from duplicate_problems(order, read, "order")


def _item_reader(question: ValuesQuestion) -> Callable[[str], Value]:
    """How each item is read: as the first value is, the values being all of one type."""
    first_value_question = question.value_question(1)
    assert first_value_question is not None
    return first_value_question.read
