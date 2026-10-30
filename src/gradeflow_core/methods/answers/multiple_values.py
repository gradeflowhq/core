from collections.abc import Iterator, Mapping
from dataclasses import replace
from itertools import permutations
from typing import Any

from pydantic import Field

from ...common.hints import ADVANCED
from ...common.paths import Loc
from ...common.problems import Problem
from ...common.text import markdown_list
from ...questions.types import QuestionId
from ...questions.values import ValuesQuestion
from ..check import Check, Context
from ..example import examples_beside
from ..logic.combination import (
    CombinationSettings,
    Counted,
    combine,
    heading,
    marked_by,
    marked_in_order,
    read_by,
)
from ..method import Method
from ..outcome import CORRECT, Outcome
from ..position import Position
from ..settings import Settings, specialised
from .accepted_answers import AcceptedAnswers

MAX_ANY_ORDER_VALUES = 6


class MultipleValuesSettings(CombinationSettings):
    any_order: bool = Field(
        default=False,
        title="Values in any order",
        description="Each check may mark any value, and the pairing that gives the most marks "
        "counts.",
        json_schema_extra=ADVANCED,
    )


class MultipleValues(Method[MultipleValuesSettings]):
    name = "gradeflow/multiple-values"
    version = 1
    label = "Multiple Values"
    summary = "Marks for each value of a multi-part answer."
    guide = (
        "**Use it when:** The answer has several values, such as a place and a year, or the roots "
        "of an equation.\n\n"
        "**How it marks:** Each value gets its own check, set up as for a single answer. **How "
        "many must be correct** sets whether all are needed, any one is enough, or each earns its "
        "share. With **Values in any order**, the pairing of values and checks that gives the "
        "most marks counts.\n\n"
        "**Good to know:** The question sets how many values there are, and there's one check "
        f"for each. Any order works for up to {MAX_ANY_ORDER_VALUES} values.\n\n"
        "**Related:** To mark items in a sequence, use Order."
    )
    question_types = frozenset({"values"})
    settings = MultipleValuesSettings
    examples = examples_beside(__file__)

    def settings_at(self, position: Position) -> type[Settings]:
        """A check for each value: no more, no fewer."""
        question = position.question_model
        if not isinstance(question, ValuesQuestion):
            return self.settings
        count = question.value_count
        return specialised(self.settings, checks={"min_length": count, "max_length": count})

    def initial_settings(self, position: Position) -> dict[str, Any]:
        question = position.question_model
        if not isinstance(question, ValuesQuestion):
            return {}
        return {
            "checks": [
                {"uses": AcceptedAnswers().ref, "with": {"answers": []}}
                for _ in range(question.value_count)
            ]
        }

    def inside(self, position: Position, loc: Loc) -> Position:
        """Each check is at the value in its place."""
        return replace(position, value=int(loc[-1]) + 1, value_fixed=True)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        values = markdown_list(
            f"**{child.name or f'Value {place}'}:** {child.describe()}"
            for place, child in enumerate(settings.checks, start=1)
        )
        return f"{heading(settings, *(['in any order'] if settings.any_order else []))}\n\n{values}"

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        assert question is not None  # it marks a values question
        settings = self._settings_of(check)
        children = settings.checks
        if not settings.any_order:
            places = range(1, len(children) + 1)
            marked = marked_in_order(children, context, question, places)
            counted: Counted = [
                (child, _at(outcomes, child, place))
                for place, (child, outcomes) in enumerate(marked, start=1)
            ]
            return _spoken(combine(settings, counted, context.questions))
        count = len(context.answers.get(question) or [])  # type: ignore[arg-type]
        table = {
            (index, place): _at(child.mark(context, question, place), child, place)
            for index, child in enumerate(children)
            for place in range(1, count + 1)
        }

        def combined(places: tuple[int, ...]) -> dict[QuestionId, Outcome]:
            counted = [
                (child, table[index, place])
                for index, (child, place) in enumerate(zip(children, places, strict=False))
            ]
            return combine(settings, counted, context.questions)

        pairings = permutations(range(1, count + 1), len(children))
        return _spoken(
            max(
                (combined(places) for places in pairings),
                key=lambda outcomes: outcomes.get(question, Outcome(0.0)).proportion,
                default={},
            )
        )

    def ordered(self, settings: Mapping[str, Any]) -> bool:
        return not settings.get("any_order", False)

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return marked_by(self._settings_of(check).checks, question)

    def questions_read(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return read_by(self._settings_of(check).checks, question)

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        question = position.question_model
        if (
            settings.any_order
            and isinstance(question, ValuesQuestion)
            and question.value_count > MAX_ANY_ORDER_VALUES
        ):
            yield Problem(
                f"Any order works for up to {MAX_ANY_ORDER_VALUES} values.", field=("any_order",)
            )


def _at(outcomes: dict[QuestionId, Outcome], check: Check, place: int) -> dict[QuestionId, Outcome]:
    """Outcomes named by the value they marked, so each line of feedback says which it is."""
    return {
        key: replace(outcome, name=check.name or f"Value {place}")
        for key, outcome in outcomes.items()
    }


def _spoken(outcomes: dict[QuestionId, Outcome]) -> dict[QuestionId, Outcome]:
    """An answer whose every value is right is “Correct.”, without a line for each."""
    return {
        key: replace(outcome, feedback=CORRECT) if outcome.correct else outcome
        for key, outcome in outcomes.items()
    }
