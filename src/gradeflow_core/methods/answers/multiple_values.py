from collections.abc import Iterator, Mapping
from dataclasses import replace
from itertools import permutations
from typing import Any

from pydantic import Field

from ...common.hints import ADVANCED_HINT
from ...common.problems import Problem
from ...common.text import markdown_list
from ...questions.types import QuestionId
from ...questions.values import ValuesQuestion
from ..example import examples_beside
from ..logic.workflow import (
    MARKS_BY_FIELD,
    MarksBy,
    NestedCheck,
    combine,
    description_heading,
    run_steps,
)
from ..method import MarkingMethod
from ..outcome import CORRECT_FEEDBACK, Outcome
from ..position import Position
from ..settings import Settings, specialized
from ..step import Check, Context, Step
from .accepted_answers import AcceptedAnswers

MAX_ANY_ORDER_VALUES = 6


class MultipleValuesSettings(Settings):
    marks_by: MarksBy = MARKS_BY_FIELD
    checks: list[NestedCheck] = Field(
        default_factory=list, title="Checks", description="One check for each value."
    )
    any_order: bool = Field(
        default=False,
        title="Values in any order",
        description="Each check may mark any value, and the pairing that gives the most marks "
        "counts.",
        json_schema_extra=ADVANCED_HINT,
    )


class MultipleValues(MarkingMethod[MultipleValuesSettings]):
    name = "gradeflow/multiple-values"
    version = 1
    label = "Multiple Values"
    summary = "Marks for each value of a multi-part answer."
    guide = (
        "**Use it when:** The answer has several values, such as a place and a year, or the roots "
        "of an equation.\n\n"
        "**How it marks:** Each value gets its own check, set up as for a single answer. **How "
        "marks are given** sets whether all are needed, any one is enough, or each earns its "
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
        value_count = question.value_count
        return specialized(
            self.settings, checks={"min_length": value_count, "max_length": value_count}
        )

    def initial_settings(self, position: Position) -> dict[str, Any]:
        question = position.question_model
        if not isinstance(question, ValuesQuestion):
            return {}
        return {
            "checks": [
                {
                    "input": f"{{{{ values[{index}] }}}}",
                    "uses": AcceptedAnswers().ref,
                    "with": {"answers": []},
                }
                for index in range(question.value_count)
            ]
        }

    def describe(self, step: Step) -> str:
        settings = self._settings_of(step)
        check_lines = markdown_list(
            f"**{check.name or f'Value {value_number}'}:** {check.describe()}"
            for value_number, check in enumerate(settings.checks, start=1)
        )
        heading = description_heading(
            settings.marks_by, *(["in any order"] if settings.any_order else [])
        )
        return f"{heading}\n{check_lines}"

    def _mark_student(
        self, settings: MultipleValuesSettings, context: Context, question_id: QuestionId | None
    ) -> dict[QuestionId, Outcome]:
        assert question_id is not None  # it marks a values question
        checks = settings.checks
        if not settings.any_order:
            [check_outcomes], _ = run_steps(checks, [context], question_id)
            named_check_outcomes = [
                (check, _named_by_value(outcomes, check, value_number))
                for value_number, (check, outcomes) in enumerate(check_outcomes, start=1)
            ]
            return _with_correct_feedback(
                combine(settings.marks_by, named_check_outcomes, context.questions)
            )
        answer = context.answers.get(question_id)
        assert isinstance(answer, list)  # it marks a values question
        value_count = len(answer)
        outcomes_by_check_and_value = {
            (index, value_number): _named_by_value(
                check.model_copy(update={"input": None}).mark(
                    [context.with_value_as_answer(question_id, value_number)], question_id
                )[0],
                check,
                value_number,
            )
            for index, check in enumerate(checks)
            for value_number in range(1, value_count + 1)
        }

        def combined_for_pairing(value_numbers: tuple[int, ...]) -> dict[QuestionId, Outcome]:
            check_outcomes = [
                (check, outcomes_by_check_and_value[index, value_number])
                for index, (check, value_number) in enumerate(
                    zip(checks, value_numbers, strict=True)
                )
            ]
            return combine(settings.marks_by, check_outcomes, context.questions)

        pairings = permutations(range(1, value_count + 1), len(checks))
        return _with_correct_feedback(
            max(
                (combined_for_pairing(value_numbers) for value_numbers in pairings),
                key=lambda outcomes: outcomes.get(question_id, Outcome(0.0)).proportion,
            )
        )

    def reads_earlier_results(self, settings: Mapping[str, Any]) -> bool:
        return not settings.get("any_order", False)

    def problems(self, step: Step, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(step)
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
    outcomes: dict[QuestionId, Outcome], check: Check, value_number: int
) -> dict[QuestionId, Outcome]:
    """Outcomes named by the value they marked, so each line of feedback says which it is."""
    if check.name:
        return {
            question_id: replace(outcome, name=check.name)
            for question_id, outcome in outcomes.items()
        }
    value_label = f"Value {value_number}"
    return {
        question_id: replace(outcome, name=value_label, labels=(*outcome.labels, value_label))
        for question_id, outcome in outcomes.items()
    }


def _with_correct_feedback(outcomes: dict[QuestionId, Outcome]) -> dict[QuestionId, Outcome]:
    """An answer whose every value is correct is “Correct.”, without a line for each."""
    return {
        question_id: replace(outcome, feedback=CORRECT_FEEDBACK) if outcome.correct else outcome
        for question_id, outcome in outcomes.items()
    }
