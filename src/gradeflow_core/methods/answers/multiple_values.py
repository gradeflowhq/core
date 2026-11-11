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
from ...steps.combining import COMBINATIONS, Part, combine
from ...steps.context import Context
from ...steps.example import examples_beside
from ...steps.method import MarkingMethod
from ...steps.nested_steps import (
    MarksBySetting,
    NestedCheck,
    combined_part,
    questions_marked_by,
    questions_read_by,
)
from ...steps.outcome import CORRECT_FEEDBACK, Outcome, QuestionOutcomes
from ...steps.position import Position
from ...steps.running import mark, run_steps
from ...steps.settings import Settings, specialized
from ...steps.step import Step
from .accepted_answers import AcceptedAnswers

MAX_ANY_ORDER_VALUES = 6


class MultipleValuesSettings(Settings):
    marks_by: MarksBySetting = "all_correct"
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
        f"for each. Any order works for up to {MAX_ANY_ORDER_VALUES} values of one type.\n\n"
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

    def describe(self, settings: MultipleValuesSettings) -> str:
        check_lines = markdown_list(
            f"**{check.name or f'Value {value_number}'}:** {check.describe()}"
            for value_number, check in enumerate(settings.checks, start=1)
        )
        heading = COMBINATIONS[settings.marks_by].description_heading(
            *(["in any order"] if settings.any_order else [])
        )
        return f"{heading}\n{check_lines}"

    def mark_student(
        self, settings: MultipleValuesSettings, context: Context, question_id: QuestionId | None
    ) -> QuestionOutcomes:
        assert question_id is not None  # it marks a values question
        checks = settings.checks
        if not settings.any_order:
            [check_outcomes], _ = run_steps(checks, [context], question_id)
            return _with_correct_feedback(
                combine(
                    settings.marks_by,
                    [
                        _named_by_value(check, outcomes, value_number)
                        for value_number, (check, outcomes) in enumerate(check_outcomes, start=1)
                    ],
                    context.questions,
                )
            )
        answer = context.answers.get(question_id)
        assert isinstance(answer, list)  # it marks a values question
        value_count = len(answer)
        outcomes_by_check_and_value = {
            (index, value_number): mark(
                check.model_copy(update={"input": None}),
                [replace(context, input=context.value(question_id, value_number))],
                question_id,
            )[0]
            for index, check in enumerate(checks)
            for value_number in range(1, value_count + 1)
        }

        def combined_for_pairing(value_numbers: tuple[int, ...]) -> QuestionOutcomes:
            return combine(
                settings.marks_by,
                [
                    _named_by_value(
                        check, outcomes_by_check_and_value[index, value_number], value_number
                    )
                    for index, (check, value_number) in enumerate(
                        zip(checks, value_numbers, strict=True)
                    )
                ],
                context.questions,
            )

        def proportion(outcomes: QuestionOutcomes) -> float:
            outcome = outcomes.get(question_id)
            return outcome.proportion if isinstance(outcome, Outcome) else 0.0

        pairings = permutations(range(1, value_count + 1), len(checks))
        return _with_correct_feedback(
            max((combined_for_pairing(value_numbers) for value_numbers in pairings), key=proportion)
        )

    def runs_inner_steps_in_order(self, settings: Mapping[str, Any]) -> bool:
        return not settings.get("any_order", False)

    def questions_marked(
        self, settings: MultipleValuesSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        return questions_marked_by(settings.checks, question_id)

    def questions_read(
        self, settings: MultipleValuesSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        return questions_read_by(settings.checks, question_id)

    def problems(self, settings: MultipleValuesSettings, position: Position) -> Iterator[Problem]:
        question = position.question_model
        if not isinstance(question, ValuesQuestion):
            return
        if len(settings.checks) != question.value_count:
            yield Problem(
                f"List {question.value_count} checks, one for each value.", field="/checks"
            )
        if not settings.any_order:
            return
        if question.value_count > MAX_ANY_ORDER_VALUES:
            yield Problem(
                f"Any order works for up to {MAX_ANY_ORDER_VALUES} values.", field="/any_order"
            )
        if len(set(question.value_types)) > 1:
            yield Problem("Any order needs values of one type.", field="/any_order")


def _named_by_value(
    check: Step, outcomes: QuestionOutcomes, value_number: int
) -> tuple[Part, QuestionOutcomes]:
    """A check and its outcomes to combine. One without a name is named by the value it marked, so
    each line of feedback says which it is."""
    value_label = f"Value {value_number}"
    part = combined_part(check, value_label)
    if check.name:
        return part, outcomes
    return part, {
        question_id: replace(outcome, labels=(*outcome.labels, value_label))
        if isinstance(outcome, Outcome)
        else outcome
        for question_id, outcome in outcomes.items()
    }


def _with_correct_feedback(outcomes: QuestionOutcomes) -> QuestionOutcomes:
    """An answer whose every value is correct is “Correct.”, without a line for each."""
    return {
        question_id: replace(outcome, feedback=CORRECT_FEEDBACK)
        if isinstance(outcome, Outcome) and outcome.correct
        else outcome
        for question_id, outcome in outcomes.items()
    }
