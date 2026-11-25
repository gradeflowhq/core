from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from pydantic import Field

from ...common.problems import Problem
from ...common.text import markdown_list
from ...questions.multipart import MultipartQuestion
from ...questions.parts import part_label
from ...questions.types import QuestionId
from ...steps.combining import COMBINATIONS, combine
from ...steps.context import Context
from ...steps.example import examples_beside
from ...steps.input_sources import AnswerPartSource, input_text
from ...steps.method import Method
from ...steps.nested_steps import (
    MarksBySetting,
    NestedCheck,
    labelled_combined_part,
    questions_marked_by,
    questions_read_by,
    with_correct_feedback,
)
from ...steps.outcome import QuestionOutcomes
from ...steps.position import Position
from ...steps.running import run_steps
from ...steps.settings import Settings, specialized
from ...steps.step import Step
from .accepted_answers import AcceptedAnswers


class PartByPartSettings(Settings):
    marks_by: MarksBySetting = "all_correct"
    checks: list[NestedCheck] = Field(
        default_factory=list, title="Checks", description="One check for each part."
    )


class PartByPart(Method[PartByPartSettings]):
    name = "gradeflow/part-by-part"
    version = 1
    label = "Part by Part"
    summary = "Marks for each part of a multi-part answer."
    guide = (
        "For an answer in named parts, such as a place and a year. The question names the "
        "parts.\n\n"
        "Each part gets its own check, set up like a check of a single answer: Accepted Answers "
        "for the place, Number Range for the year. **How marks are given** combines them:\n\n"
        "- **All must be correct**: full marks only if every part is right.\n"
        "- **Any one must be correct**: full marks if at least one part is right.\n"
        "- **Split by weight**: the question's marks are shared between the parts by their "
        "checks' weights. Equal weights give each part an equal share.\n"
        "- **Add up marks**: each check's **Marks** are what its part earns when right.\n"
        "- **None may be correct**: full marks only if no part matches its check.\n\n"
        "For several answers of one kind in any order, such as the roots of an equation, make the "
        "question a list and use Item by Item."
    )
    question_types = frozenset({"multipart"})
    settings = PartByPartSettings
    examples = examples_beside(__file__)

    def settings_model_at(self, position: Position) -> type[Settings]:
        """A check for each part: no more, no fewer."""
        question = position.question_model
        if not isinstance(question, MultipartQuestion):
            return self.settings
        part_count = len(question.parts)
        return specialized(
            self.settings, checks={"min_length": part_count, "max_length": part_count}
        )

    def initial_settings(self, position: Position) -> dict[str, Any]:
        question = position.question_model
        if not isinstance(question, MultipartQuestion):
            return {}
        return {
            "checks": [
                {
                    "input": input_text(AnswerPartSource(part.name)),
                    "uses": AcceptedAnswers().ref,
                    "with": {"answers": []},
                }
                for part in question.parts
            ]
        }

    def describe(self, settings: PartByPartSettings) -> str:
        combination = COMBINATIONS[settings.marks_by]
        weight_notes = combination.weight_notes([check.weight for check in settings.checks])
        check_lines = markdown_list(
            check.described_in_list([weight_note] if weight_note else [], _default_name(check))
            for check, weight_note in zip(settings.checks, weight_notes, strict=True)
        )
        return f"{combination.description_heading()}\n{check_lines}"

    def mark_students(
        self,
        settings: PartByPartSettings,
        contexts: Sequence[Context],
        question_id: QuestionId | None,
    ) -> list[QuestionOutcomes]:
        check_outcomes_by_student, _ = run_steps(settings.checks, contexts, question_id)
        return [
            with_correct_feedback(
                combine(
                    settings.marks_by,
                    [
                        labelled_combined_part(check, outcomes, _default_name(check))
                        for check, outcomes in check_outcomes
                    ],
                    context.questions,
                )
            )
            for context, check_outcomes in zip(contexts, check_outcomes_by_student, strict=True)
        ]

    def runs_inner_steps_in_order(self, settings: Mapping[str, Any]) -> bool:
        return True

    def questions_marked(
        self, settings: PartByPartSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        return questions_marked_by(settings.checks, question_id)

    def questions_read(
        self, settings: PartByPartSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        return questions_read_by(settings.checks, question_id)

    def problems(self, settings: PartByPartSettings, position: Position) -> Iterator[Problem]:
        yield from COMBINATIONS[settings.marks_by].weight_problems(
            [check.weight for check in settings.checks], "checks"
        )
        question = position.question_model
        if isinstance(question, MultipartQuestion) and len(settings.checks) != len(question.parts):
            yield Problem(f"List {len(question.parts)} checks, one for each part.", field="/checks")


def _default_name(check: Step) -> str | None:
    """The label of the part ``check`` reads, if it reads one."""
    return part_label(check.source.name) if isinstance(check.source, AnswerPartSource) else None
