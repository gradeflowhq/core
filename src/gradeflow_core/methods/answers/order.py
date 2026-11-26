from collections.abc import Callable, Iterator
from itertools import pairwise
from typing import Literal, cast

from pydantic import Field
from rapidfuzz.distance import LCSseq

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import code_span, extra_items_sentence, reads_better_listed
from ...questions.base import Question
from ...questions.list import ListQuestion
from ...questions.multipart import MultipartQuestion
from ...questions.types import Answer, Value
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
        json_schema_extra={"uniqueItems": True, **hint(lists_answer_items=True, item_label="item")},
    )
    marks_for: Literal["exact", "items_in_order", "pairs_in_order"] = Field(
        default="items_in_order",
        title="How marks are given",
        json_schema_extra=hint(labels=MARKS_FOR_LABELS),
    )


class Order(AnswerMethod[OrderSettings]):
    name = "gradeflow/order"
    version = 1
    label = "Order"
    summary = "Marks for putting items in the correct order."
    guide = (
        "For putting items in order, such as the stages of a process or events in time. The "
        "question is a list of text or numbers.\n\n"
        "Enter the **Correct order**, first to last, and choose **How marks are given**. With the "
        "correct order A, B, C, D:\n\n"
        "- **Items in order**: a share for each item in the longest sequence that keeps the "
        "correct order, skipping the others. B, A, C, D keeps A, C, D in order, 3 of 4, and gets "
        "three quarters of the marks.\n"
        "- **Pairs in order**: a share for each neighbouring pair kept from the correct order. B, "
        "A, C, D keeps only C, D of the 3 pairs and gets a third of the marks.\n"
        "- **Only the correct order**: all or nothing.\n\n"
        "Extra items count against the share: A, B, X, C, D has 4 of 5 in order.\n\n"
        "When the order doesn't matter, use Item by Item."
    )
    question_types = frozenset({"list"})
    settings = OrderSettings
    examples = examples_beside(__file__)

    def can_run_at(self, position: Position) -> bool:
        """On a list of text or numbers, not of items in parts."""
        question = position.question_model
        return super().can_run_at(position) and not (
            isinstance(question, ListQuestion) and question.item == "parts"
        )

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
        assert isinstance(question, ListQuestion)
        assert isinstance(answer, list)
        answer = cast(list[Value], answer)  # items aren't in parts
        correct_order = list(map(_item_reader(question), settings.order))
        # Items beyond the correct order count against the share.
        item_count = max(len(answer), len(correct_order))
        extra_count = len(answer) - len(correct_order)
        match settings.marks_for:
            case "exact":
                proportion = float(answer == correct_order)
                in_order = []
            case "items_in_order":
                items_in_order_count = LCSseq.similarity(answer, correct_order)
                proportion = items_in_order_count / item_count
                in_order = [f"{items_in_order_count} of {item_count} in order."]
            case "pairs_in_order":
                pairs_in_order_count = len(set(pairwise(answer)) & set(pairwise(correct_order)))
                proportion = pairs_in_order_count / (item_count - 1)
                in_order = [f"{pairs_in_order_count} of {item_count - 1} pairs in order."]
        if proportion >= 1:
            return Outcome(1.0, CORRECT_FEEDBACK)
        extra = [extra_items_sentence(extra_count)] if extra_count > 0 else []
        expected = f"Expected {', '.join(settings.order)}."
        return Outcome(proportion, " ".join([*in_order, *extra, expected]))

    def problems(self, settings: OrderSettings, position: Position) -> Iterator[Problem]:
        question = position.question_model
        if not isinstance(question, ListQuestion):
            return
        order = settings.order
        read = _item_reader(question)
        yield from unreadable_problems(order, read, "order")
        yield from duplicate_problems(order, read, "order")


def _item_reader(question: ListQuestion) -> Callable[[str], Value]:
    """How each item is read."""
    item_question = question.item_question()
    assert not isinstance(item_question, MultipartQuestion)  # items aren't in parts
    return item_question.read
