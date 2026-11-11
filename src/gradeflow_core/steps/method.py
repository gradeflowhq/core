"""Methods: what a step does. A method is written in Python, or composed in YAML from others."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from ..common.problems import Problem
from ..common.schemas import JsonDict, json_schema
from ..questions.base import Question
from ..questions.types import Answer, QuestionId, QuestionType, Unreadable
from .context import Context, StepInput
from .example import Example
from .names import method_ref
from .outcome import Outcome, QuestionOutcomes, Skipped, TransformResult
from .position import OutputQuestions, Position
from .settings import Settings

Level = Literal["basic", "advanced", "internal"]
"""Basic methods suit most questions, advanced ones particular cases; internal ones are only used
inside other methods."""


class Method[S: Settings = Any](ABC):
    """A way of marking: its name, what it marks, its settings and what a step with them does."""

    name: str
    version: int
    """Positive, explicitly declared revision of this method's published behavior."""
    label: str
    summary: str
    """What earns the marks, in one sentence."""
    guide: str = ""
    """Markdown in four parts: Use it when, How it marks, Good to know and Related."""
    level: Level = "basic"
    question_types: frozenset[QuestionType] = frozenset()
    """The questions, or values of an answer, it can mark."""
    marks_question_group: bool = False
    """Whether it can mark a question group."""
    examples: tuple[Example, ...] = ()
    needs_model_endpoint: bool = False
    """Whether it asks a language model."""
    marks_blank_answers: bool = False
    """Whether it marks blank answers, which otherwise get no marks before it runs."""
    describes_by_steps: bool = False
    """Whether its description is that of the steps it runs, which doesn't name it."""
    settings: type[S]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__ and (
            type(getattr(cls, "version", None)) is not int or cls.version < 1
        ):
            raise TypeError(f"{cls.__name__} must declare a positive integer version.")
        if "settings" in cls.__dict__ and "label" in cls.__dict__:
            cls.settings.model_config["title"] = cls.label
            cls.settings.model_rebuild(force=True)

    @property
    def ref(self) -> str:
        """How a step names this version of the method: ``uses: gradeflow/key-points@1``."""
        return method_ref(self.name, self.version)

    def __repr__(self) -> str:
        return f"<Method {self.ref}>"

    def gives_marks(self, settings: Mapping[str, Any]) -> bool:
        """Whether a step with ``settings``, as YAML holds them, gives marks. A step's own
        properties depend on it, so it is asked of settings that may not be valid yet."""
        return True

    def runs_inner_steps_in_order(self, settings: Mapping[str, Any]) -> bool:
        """Whether the steps in a list of ``settings``, as YAML holds them, may read the results of
        those before them. Asked of settings that may not be valid yet."""
        return False

    def settings_schema(self, position: Position) -> JsonDict:
        """The JSON schema of a step's settings, its ``with``, at ``position``: such as the
        options of the question it marks."""
        return json_schema(self.settings_model_at(position))

    def can_run_at(self, position: Position) -> bool:
        """Whether a step at ``position`` can use it."""
        if position.question_model is None:
            return self.marks_question_group
        return position.question_type in self.question_types

    def settings_model_at(self, position: Position) -> type[Settings]:
        """Its settings at ``position``, such as narrowed to a question's options."""
        return self.settings

    def initial_settings(self, position: Position) -> dict[str, Any]:
        """Settings a new step at ``position`` starts with, beyond the defaults."""
        return {}

    def settings_read_at(self, settings: Mapping[str, Any], position: Position) -> dict[str, Any]:
        """Settings as written, read as a step at ``position`` reads them."""
        return dict(settings)

    def inner_position(self, position: Position, *, is_in_list: bool) -> Position:
        """``position`` as a step in its settings sees it, in a list of them if ``is_in_list``."""
        return replace(position, is_in_list=is_in_list, is_in_workflow=False)

    @abstractmethod
    def describe(self, settings: S) -> str:
        """What a step with ``settings`` does, in a sentence or two of markdown."""

    def problems(self, settings: S, position: Position) -> Iterator[Problem]:
        """What is wrong with the settings of a step at ``position``, each at the JSON Pointer of
        its field in them, such as ``/answers/0``."""
        return iter(())

    def questions_marked(self, settings: S, question_id: QuestionId | None) -> set[QuestionId]:
        """The questions a step with ``settings`` gives marks to, reading ``question_id``."""
        return {question_id} if question_id else set()

    def questions_read(self, settings: S, question_id: QuestionId | None) -> set[QuestionId]:
        """The questions whose answers the step reads, those it marks included."""
        return self.questions_marked(settings, question_id)

    def output_question_at(
        self, settings: S, position: Position, inner_output_questions: OutputQuestions
    ) -> Question | None:
        """The question that reads the output of a step with ``settings`` reading at ``position``,
        given what the steps in its settings give; None if it gives marks."""
        return None

    def mark_students(
        self, settings: S, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> Sequence[QuestionOutcomes]:
        """What a check with ``settings`` gives each student in ``contexts`` for each question it
        marks, reading ``question_id``. Unreadable answers, and blank ones unless it
        ``marks_blank_answers``, never reach it. A method that gives marks implements it."""
        raise NotImplementedError

    def transform_students(
        self, settings: S, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> Sequence[TransformResult | Skipped]:
        """What a transform with ``settings`` gives each student in ``contexts``, reading
        ``question_id``: skipped when the step whose output it gives didn't run. A method that
        can give no marks implements it."""
        raise NotImplementedError


def _readable_input(context: Context) -> tuple[StepInput, Answer]:
    """The input of a step that reads one answer, and the answer, which marking has read before
    the method runs."""
    step_input = context.input
    assert step_input is not None
    assert not isinstance(step_input.answer, Unreadable)
    return step_input, step_input.answer


class MarkingMethod[S: Settings](Method[S]):
    """A method that marks one student at a time."""

    def mark_students(
        self, settings: S, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> list[QuestionOutcomes]:
        return [self.mark_student(settings, context, question_id) for context in contexts]

    @abstractmethod
    def mark_student(
        self, settings: S, context: Context, question_id: QuestionId | None
    ) -> QuestionOutcomes:
        """What the student of ``context`` gets for each question a step with ``settings``
        marks, reading ``question_id``."""


class AnswerMethod[S: Settings](MarkingMethod[S]):
    """A method that marks one answer: the question's, or what a step's input gives."""

    def mark_student(
        self, settings: S, context: Context, question_id: QuestionId | None
    ) -> dict[QuestionId, Outcome]:
        step_input, answer = _readable_input(context)
        return {step_input.question_id: self.mark_answer(settings, answer, step_input.question)}

    @abstractmethod
    def mark_answer(self, settings: S, answer: Answer, question: Question) -> Outcome:
        """What ``answer``, as ``question`` reads it, earns from a check with ``settings``. It
        isn't blank unless the method ``marks_blank_answers``."""


class TransformMethod[S: Settings](Method[S]):
    """A method that turns one answer into another for later steps to read, and gives no
    marks."""

    def gives_marks(self, settings: Mapping[str, Any]) -> bool:
        return False

    def transform_students(
        self, settings: S, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> list[TransformResult]:
        return [
            self.transform(settings, answer, step_input.question)
            for step_input, answer in map(_readable_input, contexts)
        ]

    def output_question_at(
        self, settings: S, position: Position, inner_output_questions: OutputQuestions
    ) -> Question | None:
        question = position.question_model
        assert question is not None  # a step reading no question gives no output
        return self.output_question(settings, question)

    @abstractmethod
    def transform(self, settings: S, answer: Answer, question: Question) -> TransformResult:
        """What ``answer``, as ``question`` reads it, turns into: another answer, or
        ``Unreadable`` with the reason it has none."""

    @abstractmethod
    def output_question(self, settings: S, question: Question) -> Question:
        """The question that reads what it gives of an answer ``question`` reads."""
