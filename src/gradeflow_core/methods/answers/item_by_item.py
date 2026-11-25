from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from itertools import accumulate, pairwise, permutations, product
from operator import getitem, itemgetter
from statistics import fmean
from typing import Any

from pydantic import Field

from ...common.problems import Problem
from ...common.text import extra_items_sentence, markdown_list
from ...questions.list import ListQuestion
from ...questions.types import QuestionId
from ...steps.combining import COMBINATIONS, Part, combine
from ...steps.context import Context, StepInput
from ...steps.example import examples_beside
from ...steps.method import Method
from ...steps.nested_steps import (
    MarksBySetting,
    NestedCheck,
    labelled_combined_part,
    questions_marked_by,
    questions_read_by,
    with_correct_feedback,
)
from ...steps.outcome import Outcome, QuestionOutcomes, Skipped, outcome_of
from ...steps.position import Position
from ...steps.running import mark
from ...steps.settings import Settings
from .accepted_answers import AcceptedAnswers

MAX_CHECKS = 6


class ItemByItemSettings(Settings):
    marks_by: MarksBySetting = "all_correct"
    checks: list[NestedCheck] = Field(
        default_factory=list,
        max_length=MAX_CHECKS,
        title="Checks",
        description="One check for each item expected.",
    )


class ItemByItem(Method[ItemByItemSettings]):
    name = "gradeflow/item-by-item"
    version = 1
    label = "Item by Item"
    summary = "Marks for each item of a list, in any order."
    guide = (
        "For a list of items of one kind in any order, such as the noble gases or the roots of an "
        "equation. For items in parts, such as a problem and its fix, give each check Part by "
        "Part.\n\n"
        "Give one check for each item you expect. Items and checks are paired in the way that "
        "gives the most marks. **How marks are given** combines the checks, as in Part by "
        "Part.\n\n"
        "A check left without an item marks a blank answer. Each item beyond the checks counts as "
        "a wrong one, worth as much as an average check. Under **Add up marks** an extra item adds "
        "nothing.\n\n"
        f"It takes up to {MAX_CHECKS} checks. To mark the order of the items, use Order."
    )
    question_types = frozenset({"list"})
    settings = ItemByItemSettings
    examples = examples_beside(__file__)

    def inner_position(self, position: Position, *, is_in_list: bool) -> Position:
        """A check of one item of the list."""
        inner_position = super().inner_position(position, is_in_list=is_in_list)
        question = position.question_model
        if not isinstance(question, ListQuestion):
            return inner_position
        return replace(
            inner_position, input_question=question.item_question(), is_in_item_check=True
        )

    def initial_settings(self, position: Position) -> dict[str, Any]:
        question = position.question_model
        if not isinstance(question, ListQuestion) or question.item == "parts":
            return {}
        return {"checks": [{"uses": AcceptedAnswers().ref, "with": {"answers": []}}]}

    def describe(self, settings: ItemByItemSettings) -> str:
        combination = COMBINATIONS[settings.marks_by]
        weight_notes = combination.weight_notes([check.weight for check in settings.checks])
        check_lines = markdown_list(
            check.described_in_list([weight_note] if weight_note else [])
            for check, weight_note in zip(settings.checks, weight_notes, strict=True)
        )
        return f"{combination.description_heading('in any order')}\n{check_lines}"

    def mark_students(
        self,
        settings: ItemByItemSettings,
        contexts: Sequence[Context],
        question_id: QuestionId | None,
    ) -> list[QuestionOutcomes]:
        assert question_id is not None  # it marks a list question
        check_count = len(settings.checks)
        list_inputs = [_list_input(context) for context in contexts]
        item_counts = [item_count for _, item_count in list_inputs]
        # A student with fewer items than checks has a blank one for the other checks to read.
        item_counts_with_blank = [
            item_count + (item_count < check_count) for item_count in item_counts
        ]
        item_contexts = [
            replace(context, input=list_input.item(item_index))
            for context, (list_input, _), count in zip(
                contexts, list_inputs, item_counts_with_blank, strict=True
            )
            for item_index in range(count)
        ]
        # Each check marks every student's items at once, so a check that asks a language model
        # asks about all of them together.
        outcomes_by_check = [mark(check, item_contexts, question_id) for check in settings.checks]
        item_ranges = pairwise(accumulate(item_counts_with_blank, initial=0))
        return [
            _best_pairing(
                settings,
                context,
                question_id,
                [outcomes[start:end] for outcomes in outcomes_by_check],
                item_count,
            )
            for context, item_count, (start, end) in zip(
                contexts, item_counts, item_ranges, strict=True
            )
        ]

    def questions_marked(
        self, settings: ItemByItemSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        return questions_marked_by(settings.checks, question_id)

    def questions_read(
        self, settings: ItemByItemSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        return questions_read_by(settings.checks, question_id)

    def problems(self, settings: ItemByItemSettings, position: Position) -> Iterator[Problem]:
        return COMBINATIONS[settings.marks_by].weight_problems(
            [check.weight for check in settings.checks], "checks"
        )


def _list_input(context: Context) -> tuple[StepInput, int]:
    """The list the student's context reads, and how many items it has."""
    list_input = context.input
    assert list_input is not None  # it reads the list it marks
    assert isinstance(list_input.answer, list)  # blank and unreadable answers don't reach it
    return list_input, len(list_input.answer)


def _best_pairing(
    settings: ItemByItemSettings,
    context: Context,
    question_id: QuestionId,
    outcomes_by_check: Sequence[Sequence[QuestionOutcomes]],
    item_count: int,
) -> QuestionOutcomes:
    """The student's outcomes under the pairing of checks with items that gives the most marks,
    and on a tie the one whose checks match their items best. ``outcomes_by_check`` holds each
    check's outcome for each item, then for the blank item if there is one."""
    weights = [check.weight for check in settings.checks]
    extra_part = _extra_part(weights, item_count - len(weights), question_id)

    def combined(pairing: tuple[int, ...]) -> QuestionOutcomes:
        # Feedback follows the student's items, then the blank ones for checks left without one.
        check_items = sorted(enumerate(pairing), key=itemgetter(1))
        parts = [
            labelled_combined_part(
                settings.checks[check_index],
                outcomes_by_check[check_index][item_index],
                f"Item {item_index + 1 if item_index < item_count else number}",
            )
            for number, (check_index, item_index) in enumerate(check_items, 1)
        ]
        all_parts = [*parts, *([extra_part] if extra_part else [])]
        return combine(settings.marks_by, all_parts, context.questions)

    proportions = [
        [_proportion(outcomes, question_id) for outcomes in check_outcomes]
        for check_outcomes in outcomes_by_check
    ]

    def marks_then_matches(pairing: tuple[int, ...]) -> tuple[float, float]:
        matches = sum(map(getitem, proportions, pairing, strict=True))
        return _proportion(combined(pairing), question_id), matches

    pairings = _pairings(proportions, item_count, settings.marks_by == "none_correct")
    return with_correct_feedback(combined(max(pairings, key=marks_then_matches)))


def _proportion(outcomes: QuestionOutcomes, question_id: QuestionId) -> float:
    return outcome_of(outcomes.get(question_id, Skipped())).proportion


def _pairings(
    proportions: Sequence[Sequence[float]], item_count: int, none_may_be_correct: bool
) -> Iterable[tuple[int, ...]]:
    """The item each check reads, for every pairing worth trying. With fewer items than checks,
    the checks left over read the blank item after them. With more, each check reads one of its
    best items, as many as there are checks: one of them is always free for it and gives it no
    less."""
    check_count = len(proportions)
    if item_count < check_count:
        return set(permutations([*range(item_count), *[item_count] * (check_count - item_count)]))
    best_items = [
        sorted(
            range(item_count), key=check_proportions.__getitem__, reverse=not none_may_be_correct
        )[:check_count]
        for check_proportions in proportions
    ]
    return (pairing for pairing in product(*best_items) if len(set(pairing)) == check_count)


def _extra_part(
    weights: Sequence[float], extra_count: int, question_id: QuestionId
) -> tuple[Part, QuestionOutcomes] | None:
    """The items beyond the checks, as one wrong part worth an average check for each."""
    positive_weights = [weight for weight in weights if weight > 0]
    if extra_count <= 0 or not positive_weights:
        return None
    feedback = extra_items_sentence(extra_count)
    part = Part(extra_count * fmean(positive_weights), describe=lambda: feedback)
    return part, {question_id: Outcome(0.0, feedback)}
