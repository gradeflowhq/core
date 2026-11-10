"""Methods: what a step does. A method is written in Python, or composed in YAML from others."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field, create_model

from ..common.paths import Location
from ..common.problems import Problem
from ..common.schemas import JsonDict, json_schema
from ..questions.base import Question
from ..questions.types import Answer, QuestionId, QuestionType, Unreadable
from .example import Example
from .names import method_ref
from .outcome import MethodResult, Outcome, QuestionOutcomes, TransformResult
from .position import Position
from .settings import Settings, literal_type
from .step import STEP_OWN_FIELDS, Check, Context, Step, Transform, step_cardinality

Level = Literal["basic", "advanced", "internal"]

"""Basic methods suit most questions, advanced ones particular cases; internal ones are only used
inside other methods."""


@dataclass(frozen=True)
class Progress:
    """How far a step of marking has got, such as 12 of 30 requests to a language model."""

    completed: int
    total: int
    step: str


class Method[S: Settings, R: MethodResult = MethodResult](ABC):
    """What a step does: its name, and how people choose and understand it."""

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
    marks_blank_answers: bool = False
    """Whether it marks blank answers, which otherwise get no marks before it runs."""
    settings: type[S]
    examples: tuple[Example, ...] = ()
    needs_model_endpoint: bool = False
    """Whether it asks a language model."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__ and (
            type(getattr(cls, "version", None)) is not int or cls.version < 1
        ):
            raise TypeError(f"{cls.__name__} must declare a positive integer version.")
        if "settings" in cls.__dict__ and (
            shared_keys := STEP_OWN_FIELDS.keys() & cls.settings.model_fields
        ):
            raise TypeError(f"{cls.__name__}'s settings share a step's keys: {sorted(shared_keys)}")
        if "settings" in cls.__dict__ and "label" in cls.__dict__:
            cls.settings.model_config["title"] = cls.label
            cls.settings.model_rebuild(force=True)

    def gives_marks(self, settings: Mapping[str, Any]) -> bool:
        """Whether a step with ``settings``, as YAML holds them, gives marks."""
        return True

    @property
    def ref(self) -> str:
        """How a step names this version of the method: ``uses: gradeflow/key-points@1``."""
        return method_ref(self.name, self.version)

    def __repr__(self) -> str:
        return f"<Method {self.ref}>"

    def settings_schema(self, position: Position) -> JsonDict:
        """The JSON schema of a step's settings, its ``with``, at ``position``: such as the
        options of the question it marks."""
        return json_schema(self.settings_model_at(position))

    @cached_property
    def check_model(self) -> type[Check]:
        """The model of its steps that give marks."""
        return self._build_step_model(Check)

    @cached_property
    def transform_model(self) -> type[Transform]:
        """The model of its steps that give a value."""
        return self._build_step_model(Transform)

    def _build_step_model[T: Step](self, base: type[T]) -> type[T]:
        has_required_setting = any(
            field_info.is_required() for field_info in self.settings.model_fields.values()
        )
        with_field = (
            Field(alias="with", title=self.label)
            if has_required_setting
            else Field(default_factory=dict, validate_default=True, alias="with", title=self.label)
        )
        model = create_model(
            "".join(part.capitalize() for part in self.name.split("/")[-1].split("-")) + "Step",
            __base__=base,
            __config__=ConfigDict(title=self.label),
            uses=(literal_type([self.ref]), Field(title="Method")),
            with_=(self.settings, with_field),
        )
        model.method = self
        return model

    @abstractmethod
    def run(
        self, step: Step, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> Sequence[R]:
        """What the step gives each student in ``contexts``, reading ``question_id``: outcomes by
        question if it gives marks, or else what it read. Unreadable answers, and blank ones
        unless it ``marks_blank_answers``, are dealt with before."""

    def _settings_of(self, step: Step) -> S:
        return cast(S, step.with_)

    def can_run_at(self, position: Position) -> bool:
        """Whether a step at ``position`` can use it."""
        if position.question_model is None:
            return self.marks_question_group
        return position.question_type in self.question_types

    def output_question(
        self,
        settings: Mapping[str, Any],
        position: Position,
        methods: Mapping[str, Method[Any]],
    ) -> Question | None:
        """The question that reads the output of a step with ``settings``, as YAML holds them,
        reading at ``position``; None if it gives marks."""
        return None

    def settings_model_at(self, position: Position) -> type[Settings]:
        """Its settings at ``position``, such as narrowed to a question's options."""
        return self.settings

    def initial_settings(self, position: Position) -> dict[str, Any]:
        """Settings a new step at ``position`` starts with, beyond the defaults."""
        return {}

    def settings_read_at(self, settings: Mapping[str, Any], position: Position) -> dict[str, Any]:
        """Settings as written, read as a step at ``position`` reads them."""
        return dict(settings)

    @abstractmethod
    def describe(self, step: Step) -> str:
        """What the step does, in a sentence or two of markdown."""

    @cached_property
    def _holds_steps(self) -> bool:
        """Whether its settings hold steps."""
        return any(
            step_cardinality(field_info) for field_info in self.settings.model_fields.values()
        )

    def questions_marked(self, step: Step, question_id: QuestionId | None) -> set[QuestionId]:
        """The questions the step gives marks to."""
        if not self._holds_steps:
            return {question_id} if question_id else set()
        return {
            marked_question_id
            for _, inner_step in self.expanded_steps(step)
            if inner_step.has_weight
            for marked_question_id in inner_step.questions_marked(question_id)
        }

    def questions_read(self, step: Step, question_id: QuestionId | None) -> set[QuestionId]:
        """The questions whose answers the step reads, those it marks included."""
        if not self._holds_steps:
            return self.questions_marked(step, question_id)
        return {
            read_question_id
            for _, inner_step in self.expanded_steps(step)
            for read_question_id in inner_step.questions_read(question_id)
        }

    def inner_steps(self, step: Step) -> Iterator[tuple[Location, Step]]:
        """The steps written in its settings, and where they are."""
        settings = self._settings_of(step)
        for name, field_info in type(settings).model_fields.items():
            value = getattr(settings, name)
            match step_cardinality(field_info):
                case "one" if isinstance(value, Step):
                    yield ("with", name), value
                case "many":
                    for index, item in enumerate(value):
                        yield ("with", name, index), item

    def expanded_steps(self, step: Step) -> Iterator[tuple[Location, Step]]:
        """Its inner steps as they run: those a method file expands to."""
        yield from self.inner_steps(step)

    def reads_earlier_results(self, settings: Mapping[str, Any]) -> bool:
        """Whether inner steps in a list may read the results of those before them."""
        return False

    def inner_position(self, position: Position, location: Location) -> Position:
        """``position`` as a step at ``location`` in its settings sees it."""
        return replace(position, is_in_list=isinstance(location[-1], int), is_in_workflow=False)

    def problems(self, step: Step, position: Position) -> Iterator[Problem]:
        """What is wrong with the step's own settings at ``position``, each at its setting."""
        return iter(())


class MarkingMethod[S: Settings](Method[S, QuestionOutcomes]):
    """A method that marks one student at a time."""

    def run(
        self, step: Step, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> list[QuestionOutcomes]:
        settings = self._settings_of(step)
        return [self._mark_student(settings, context, question_id) for context in contexts]

    @abstractmethod
    def _mark_student(
        self, settings: S, context: Context, question_id: QuestionId | None
    ) -> QuestionOutcomes: ...


class AnswerMethod[S: Settings](MarkingMethod[S]):
    """A method that marks one answer: the question's, or what a step's input gives."""

    def _mark_student(
        self, settings: S, context: Context, question_id: QuestionId | None
    ) -> QuestionOutcomes:
        assert question_id is not None  # a check that marks an answer has a question
        answer = context.answers.get(question_id)
        assert not isinstance(answer, Unreadable)  # marking deals with these first
        return {question_id: self._mark_answer(settings, answer, context.questions[question_id])}

    @abstractmethod
    def _mark_answer(self, settings: S, answer: Answer, question: Question) -> Outcome: ...


class TransformMethod[S: Settings](Method[S, TransformResult]):
    """A method that turns one answer into another for later steps to read, and gives no
    marks."""

    def run(
        self, step: Step, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> list[TransformResult]:
        assert question_id is not None  # a transform reads one question's answer
        settings = self._settings_of(step)
        results = []
        for context in contexts:
            answer = context.answers.get(question_id)
            assert not isinstance(answer, Unreadable)  # marking deals with these first
            results.append(self._transform(settings, answer, context.questions[question_id]))
        return results

    @abstractmethod
    def _transform(self, settings: S, answer: Answer, question: Question) -> TransformResult:
        """What ``answer``, read by ``question``, turns into."""

    def gives_marks(self, settings: Mapping[str, Any]) -> bool:
        return False

    @abstractmethod
    def output_question(
        self,
        settings: Mapping[str, Any],
        position: Position,
        methods: Mapping[str, Method[Any]],
    ) -> Question: ...
