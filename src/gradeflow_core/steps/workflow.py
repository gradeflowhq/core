from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, field_validator

from ..common.hints import hint
from ..common.problems import Problem
from ..common.text import markdown_list
from ..questions.base import Question
from ..questions.question_set import QuestionSet
from ..questions.types import ALL_QUESTION_TYPES, QuestionId
from .combining import COMBINATIONS, MARKS_BY_LABELS, MarksBy, combine
from .context import Context
from .example import examples_beside
from .input_sources import StepOutputSource, input_source
from .method import Method
from .nested_steps import (
    MarksBySetting,
    NestedStep,
    combined_part,
    questions_marked_by,
    questions_read_by,
)
from .outcome import Outcome, QuestionOutcomes, Skipped, TransformResult
from .position import MARKS_AS_ANSWER_MESSAGE, OutputQuestions, Position
from .running import CheckOutcomes, run_steps
from .settings import Settings, specialized
from .step import Step
from .templates import names_read_in

NO_STEPS_PROBLEM = Problem("Add at least one step.", field=("steps",), is_incomplete=True)


class WorkflowSettings(Settings):
    output: Literal["marks", "step_output"] = Field(
        default="marks",
        title="Output",
        json_schema_extra=hint(labels={"marks": "Marks", "step_output": "A step's output"}),
    )
    marks_by: MarksBySetting = Field(
        default="all_correct",
        json_schema_extra=hint(labels=MARKS_BY_LABELS, applies_when={"output": ["marks"]}),
    )
    output_step: str | None = Field(
        default=None,
        title="Step",
        description="The step whose output this gives.",
        json_schema_extra=hint(kind="output_step", applies_when={"output": ["step_output"]}),
    )
    steps: list[NestedStep] = Field(
        default_factory=list,
        title="Steps",
        json_schema_extra=hint(item_label="step"),
    )

    @field_validator("output_step")
    @classmethod
    def _validate_output_step(cls, output_step: str | None) -> str | None:
        if output_step is not None and _output_step_id(output_step) is None:
            raise ValueError("Name a step, such as {{ steps.name.output }}.")
        return output_step

    @property
    def output_step_id(self) -> str | None:
        """The ID of the step whose output it gives."""
        return _output_step_id(self.output_step) if self.output_step else None


def _output_step_id(output_step: str) -> str | None:
    """The ID of the step ``output_step`` names, if it names one."""
    try:
        source = input_source(output_step)
    except ValueError:
        return None
    return source.step_id if isinstance(source, StepOutputSource) else None


def _problems_of_steps(
    steps: Sequence[Step], step_ids_read_by_output: frozenset[str] = frozenset()
) -> Iterator[Problem]:
    """What is wrong with each transform among ``steps``: one no later step can read, or reads."""
    for index, step in enumerate(steps):
        if step.resolution.gives_marks:
            continue
        later_steps = [later_step.dump() for later_step in steps[index + 1 :]]
        if step.id is None:
            yield Problem("Give it an ID.", field=("steps", index, "id"))
        elif step.id not in names_read_in(later_steps, "steps") | step_ids_read_by_output:
            yield Problem(
                f"No later step reads {step.method_label}.", field=("steps",), is_incomplete=True
            )


class Workflow(Method[WorkflowSettings]):
    name = "gradeflow/workflow"
    version = 1
    label = "Workflow"
    summary = "Several steps combined into one mark."
    guide = (
        "**Use it when:** One check isn't enough, such as an answer with reasoning and a final "
        "number, or an answer that must name a key term and stay within a word limit.\n\n"
        "**How it marks:** Steps run in order. **How marks are given** sets how the checks count: "
        "**All must be correct** and **None may be correct** give full marks or none, **Any one "
        "must be correct** counts the check that gives the most marks, **Total** adds up the "
        "correct checks by their weights, as marks of the question, and **Proportional** gives "
        "each its share of the marks by its weight. A transform gives no marks: it finds part of "
        "the answer for a later step to read as its **Input**.\n\n"
        "**Good to know:** A weight below 1, such as `0.5`, makes a check worth part of the marks. "
        "A weight of `0` lets later steps read a check without it giving marks, and a negative "
        "weight is a penalty. With **Any one must be correct**, list the correct answer first, "
        "then partly correct answers with lower weights.\n\n"
        "**Related:** To forbid something, add a Not Allowed check. To mark several questions "
        "together, make a question group."
    )
    level = "advanced"
    question_types = ALL_QUESTION_TYPES
    marks_question_group = True
    settings = WorkflowSettings
    examples = examples_beside(__file__)

    def gives_marks(self, settings: Mapping[str, Any]) -> bool:
        return bool(settings.get("output", "marks") == "marks")

    def runs_inner_steps_in_order(self, settings: Mapping[str, Any]) -> bool:
        return True

    def settings_model_at(self, position: Position) -> type[Settings]:
        """Outside a workflow, nothing can read a step's output."""
        if position.is_in_workflow:
            return self.settings
        return specialized(
            self.settings,
            excluded_fields=("output", "output_step"),
            marks_by={"json_schema_extra": hint(labels=MARKS_BY_LABELS)},
        )

    def inner_position(self, position: Position, *, is_in_list: bool) -> Position:
        return replace(position, is_in_list=is_in_list, is_in_workflow=True)

    def describe(self, settings: WorkflowSettings) -> str:
        if settings.output == "step_output":
            return "\n".join(inner_step.describe() for inner_step in settings.steps)
        combination = COMBINATIONS[settings.marks_by]
        step_list_markdown = markdown_list(
            inner_step.described_in_list(combination.weight_notes(inner_step.weight))
            for inner_step in settings.steps
        )
        return f"{combination.description_heading()}\n{step_list_markdown}"

    def questions_marked(
        self, settings: WorkflowSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        return questions_marked_by(settings.steps, question_id)

    def questions_read(
        self, settings: WorkflowSettings, question_id: QuestionId | None
    ) -> set[QuestionId]:
        return questions_read_by(settings.steps, question_id)

    def output_question_at(
        self,
        settings: WorkflowSettings,
        position: Position,
        inner_output_questions: OutputQuestions,
    ) -> Question | None:
        """The question reading the output of the step it names, or else its own answer."""
        return inner_output_questions.get(settings.output_step_id or "") or position.question_model

    def mark_students(
        self,
        settings: WorkflowSettings,
        contexts: Sequence[Context],
        question_id: QuestionId | None,
    ) -> list[QuestionOutcomes]:
        check_outcomes_by_student, _ = run_steps(settings.steps, contexts, question_id)
        return [
            _combined_outcomes_for_student(
                settings.marks_by, student_check_outcomes, context.questions
            )
            for context, student_check_outcomes in zip(
                contexts, check_outcomes_by_student, strict=True
            )
        ]

    def transform_students(
        self,
        settings: WorkflowSettings,
        contexts: Sequence[Context],
        question_id: QuestionId | None,
    ) -> list[TransformResult | Skipped]:
        _, final_contexts = run_steps(settings.steps, contexts, question_id)
        return [
            result.transform_result
            if (result := context.earlier_results.get(settings.output_step_id or ""))
            else Skipped()
            for context in final_contexts
        ]

    def problems(self, settings: WorkflowSettings, position: Position) -> Iterator[Problem]:
        steps = settings.steps
        if not steps:
            yield NO_STEPS_PROBLEM
        if settings.output == "step_output":
            yield from self._output_step_problems(settings)
            return
        if steps and not any(inner_step.weight != 0 for inner_step in steps):
            yield Problem("Add a step that gives marks.", field=("steps",), is_incomplete=True)
        yield from _problems_of_steps(steps)
        yield from COMBINATIONS[settings.marks_by].weight_problems(
            [inner_step.weight for inner_step in steps], "steps"
        )

    def _output_step_problems(self, settings: WorkflowSettings) -> Iterator[Problem]:
        step_id = settings.output_step_id
        output_step = next(
            (inner_step for inner_step in settings.steps if inner_step.id == step_id), None
        )
        if step_id is None:
            yield Problem(
                "Choose the step whose output it gives.", field=("output_step",), is_incomplete=True
            )
        elif output_step is None:
            yield Problem(f"No step here is called {step_id}.", field=("output_step",))
        elif output_step.resolution.gives_marks:
            yield Problem(MARKS_AS_ANSWER_MESSAGE, field=("output_step",))
        yield from _problems_of_steps(settings.steps, frozenset({step_id} if step_id else ()))


def _combined_outcomes_for_student(
    marks_by: MarksBy, check_outcomes: CheckOutcomes, questions: QuestionSet
) -> QuestionOutcomes:
    combined_outcomes = combine(
        marks_by,
        [(combined_part(check), outcomes) for check, outcomes in check_outcomes],
        questions,
    )
    # A check reading a transform keeps its row, which says what it read.
    if len(check_outcomes) > 1 or any(check.input for check, _ in check_outcomes):
        return combined_outcomes
    return {
        question_id: _as_only_check(outcome) for question_id, outcome in combined_outcomes.items()
    }


def _as_only_check(outcome: Outcome | Skipped) -> Outcome | Skipped:
    """A workflow of one check, replaced by its only child."""
    if isinstance(outcome, Skipped) or len(outcome.children) != 1:
        return outcome
    [only_child] = outcome.children
    return replace(
        outcome,
        title=only_child.outcome.title,
        labels=only_child.outcome.labels,
        details=only_child.outcome.details,
        children=only_child.outcome.children,
    )
