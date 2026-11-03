from collections.abc import Iterator, Mapping
from dataclasses import replace
from itertools import permutations
from typing import Any

from pydantic import Field

from ...common.hints import ADVANCED_HINT
from ...common.paths import Location
from ...common.problems import Problem
from ...common.text import markdown_list
from ...questions.types import QuestionId
from ...questions.values import ValuesQuestion
from ..check import Check, Context
from ..example import examples_beside
from ..logic.combination import (
    CheckOutcomes,
    CombinationSettings,
    combine,
    description_heading,
    outcomes_in_order,
    questions_marked_by,
    questions_read_by,
)
from ..method import Method
from ..outcome import CORRECT, Outcome
from ..position import Position
from ..settings import Settings, specialized
from .accepted_answers import AcceptedAnswers

MAX_ANY_ORDER_VALUES = 6


class MultipleValuesSettings(CombinationSettings):
    any_order: bool = Field(
        default=False,
        title="Values in any order",
        description="Each check may mark any value, and the pairing that gives the most marks "
        "counts.",
        json_schema_extra=ADVANCED_HINT,
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

    def settings_model_at(self, position: Position) -> type[Settings]:
        """A check for each value: no more, no fewer."""
        question = position.question_model
        if not isinstance(question, ValuesQuestion):
            return self.settings
        count = question.value_count
        return specialized(self.settings, checks={"min_length": count, "max_length": count})

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

    def inner_position(self, position: Position, location: Location) -> Position:
        """Each check is at the value in its place."""
        return replace(position, value=int(location[-1]) + 1, value_fixed=True)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        values = markdown_list(
            f"**{inner_check.name or f'Value {place}'}:** {inner_check.describe()}"
            for place, inner_check in enumerate(settings.checks, start=1)
        )
        heading = description_heading(settings, *(["in any order"] if settings.any_order else []))
        return f"{heading}\n{values}"

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        assert question is not None  # it marks a values question
        settings = self._settings_of(check)
        checks = settings.checks
        if not settings.any_order:
            places = range(1, len(checks) + 1)
            check_outcomes: CheckOutcomes = [
                (inner_check, _named_by_value(outcomes, inner_check, place))
                for place, (inner_check, outcomes) in enumerate(
                    outcomes_in_order(checks, context, question, places), start=1
                )
            ]
            return _with_correct_feedback(combine(settings, check_outcomes, context.questions))
        count = len(context.answers.get(question) or [])  # type: ignore[arg-type]
        outcomes_by_check_and_place = {
            (index, place): _named_by_value(
                inner_check.mark(context, question, place), inner_check, place
            )
            for index, inner_check in enumerate(checks)
            for place in range(1, count + 1)
        }

        def combined_for_pairing(places: tuple[int, ...]) -> dict[QuestionId, Outcome]:
            check_outcomes = [
                (inner_check, outcomes_by_check_and_place[index, place])
                for index, (inner_check, place) in enumerate(zip(checks, places, strict=False))
            ]
            return combine(settings, check_outcomes, context.questions)

        pairings = permutations(range(1, count + 1), len(checks))
        return _with_correct_feedback(
            max(
                (combined_for_pairing(places) for places in pairings),
                key=lambda outcomes: outcomes.get(question, Outcome(0.0)).proportion,
                default={},
            )
        )

    def reads_earlier_results(self, settings: Mapping[str, Any]) -> bool:
        return not settings.get("any_order", False)

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return questions_marked_by(self._settings_of(check).checks, question)

    def questions_read(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return questions_read_by(self._settings_of(check).checks, question)

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        question = position.question_model
        if not isinstance(question, ValuesQuestion):
            return
        if len(settings.checks) != question.value_count:
            yield Problem(
                f"List {question.value_count} checks, one for each value.", field=("checks",)
            )
        if settings.any_order and question.value_count > MAX_ANY_ORDER_VALUES:
            yield Problem(
                f"Any order works for up to {MAX_ANY_ORDER_VALUES} values.", field=("any_order",)
            )


def _named_by_value(
    outcomes: dict[QuestionId, Outcome], check: Check, place: int
) -> dict[QuestionId, Outcome]:
    """Outcomes named by the value they marked, so each line of feedback says which it is."""
    if check.name:
        return {key: replace(outcome, name=check.name) for key, outcome in outcomes.items()}
    value = f"Value {place}"
    return {
        key: replace(outcome, name=value, labels=(*outcome.labels, value))
        for key, outcome in outcomes.items()
    }


def _with_correct_feedback(outcomes: dict[QuestionId, Outcome]) -> dict[QuestionId, Outcome]:
    """An answer whose every value is correct is “Correct.”, without a line for each."""
    return {
        key: replace(outcome, feedback=CORRECT) if outcome.correct else outcome
        for key, outcome in outcomes.items()
    }
