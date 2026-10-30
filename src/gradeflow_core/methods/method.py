"""Methods: kinds of checks. A method is written in Python, or composed in YAML from others."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, cast

from pydantic import Field, create_model

from ..common.model_service import ModelEndpoint
from ..common.paths import Loc
from ..common.problems import Problem
from ..common.schemas import JsonDict, json_schema
from ..questions.base import Question
from ..questions.question_set import read_answer
from ..questions.types import Answer, AnswerShape, QuestionId, QuestionType, Unreadable
from .check import CHECK_FIELDS, Check, Context, holds_checks
from .example import Example
from .names import method_ref
from .outcome import Outcome
from .position import Position
from .settings import Settings, literal

Level = Literal["basic", "advanced", "internal"]

"""Basic methods suit most questions, advanced ones particular cases; internal ones are only used
inside other methods."""


@dataclass(frozen=True)
class Progress:
    """How far a step of marking has got, such as 12 of 30 requests to a language model."""

    done: int
    total: int
    step: str


class Method[S: Settings](ABC):
    """A kind of check: its name, how people choose and understand it, and how it marks. What
    the README shows is for using methods; its other members are for writing them."""

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
    answer_shape: AnswerShape | Literal["both"] = "single"
    """Whether it marks one answer, an answer read as several possible answers, or both."""
    group: bool = False
    """Whether it can mark a question group."""
    marks_blank: bool = False
    """Whether it marks blank answers, which otherwise get no marks before it runs."""
    settings: type[S]
    examples: tuple[Example, ...] = ()
    _check_base: type[Check] = Check
    """What the model of its checks is built on."""
    needs_model_endpoint: bool = False
    """Whether it asks a language model, through the endpoint ``mark`` is given."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "name" in cls.__dict__ and (
            type(getattr(cls, "version", None)) is not int or cls.version < 1
        ):
            raise TypeError(f"{cls.__name__} must declare a positive integer version.")
        if "settings" in cls.__dict__ and (
            shared := CHECK_FIELDS.keys() & cls.settings.model_fields
        ):
            raise TypeError(f"{cls.__name__}'s settings share a check's keys: {sorted(shared)}")
        if "settings" in cls.__dict__ and "label" in cls.__dict__:
            cls.settings.model_config["title"] = cls.label
            cls.settings.model_rebuild(force=True)

    @property
    def ref(self) -> str:
        """How a check names this version of the method: ``uses: gradeflow/key-points@1``."""
        return method_ref(self.name, self.version)

    def __repr__(self) -> str:
        return f"<Method {self.ref}>"

    @property
    def setting_titles(self) -> list[str]:
        """What a check of it is set up with, by title, in order."""
        return [info.title or name for name, info in self.settings.model_fields.items()]

    def settings_schema(self, position: Position) -> JsonDict:
        """The JSON schema of a check's settings, its ``with``, at ``position``: such as the
        options of the question it marks."""
        return json_schema(self.settings_at(position))

    @cached_property
    def check_model(self) -> type[Check]:
        required = any(field.is_required() for field in self.settings.model_fields.values())
        settings = (
            Field(alias="with", title=self.label)
            if required
            else Field(default_factory=dict, validate_default=True, alias="with", title=self.label)
        )
        model = create_model(
            "".join(part.capitalize() for part in self.name.split("/")[-1].split("-")) + "Check",
            __base__=self._check_base,
            uses=(literal([self.ref]), Field(title="Method")),
            with_=(self.settings, settings),
        )
        model.method = self
        model.model_config["title"] = self.label
        return model

    def _settings_of(self, check: Check) -> S:
        return cast(S, check.with_)

    def fits(self, position: Position) -> bool:
        if position.question is None:
            return self.group
        if position.value is not None and self.needs_model_endpoint:
            return False  # language models are asked about whole answers, once for everyone's
        shapes = {"both", position.answer_shape}
        return position.answer_type in self.question_types and self.answer_shape in shapes

    def settings_at(self, position: Position) -> type[Settings]:
        """Its settings at ``position``, such as narrowed to a question's options."""
        return self.settings

    def initial_settings(self, position: Position) -> dict[str, Any]:
        """Settings a new check at ``position`` starts with, beyond the defaults."""
        return {}

    def read_at(self, settings: Mapping[str, Any], position: Position) -> dict[str, Any]:
        """Settings as written, read as a check at ``position`` reads them."""
        return dict(settings)

    @abstractmethod
    def describe(self, check: Check) -> str:
        """What the check does, in a sentence or two of markdown."""

    @abstractmethod
    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        """The outcome for each question the check marks, reading ``question`` or its ``value``.
        Unreadable answers, and blank ones unless it ``marks_blank``, are dealt with before."""

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        """The questions the check gives marks to."""
        return {question} if question else set()

    def questions_read(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        """The questions whose answers the check reads, those it marks included."""
        return self.questions_marked(check, question)

    def nested(self, check: Check) -> Iterator[tuple[Loc, Check, bool]]:
        """The checks in its settings, where they are, and whether they are in a list."""
        settings = self._settings_of(check)
        for name, info in type(settings).model_fields.items():
            held = holds_checks(info)
            value = getattr(settings, name)
            if held == "one" and isinstance(value, Check):
                yield ("with", name), value, False
            elif held == "many":
                for index, item in enumerate(value):
                    yield ("with", name, index), item, True

    def validation_children(self, check: Check) -> Iterator[tuple[Loc, Check, bool]]:
        """Checks to validate, including generated bodies that are not editable settings."""
        yield from self.nested(check)

    def ordered(self, settings: Mapping[str, Any]) -> bool:
        """Whether children in a list may read preceding sibling results."""
        return False

    def inside(self, position: Position, loc: Loc) -> Position:
        """Where the check at ``loc`` of its settings reads, when its place there decides more
        than ``position``, such as the value it reads."""
        return position

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        """What is wrong with the check's own settings at ``position``, each at its setting."""
        return iter(())

    def precompute(
        self,
        checks: Sequence[Check],
        answers: Sequence[str],
        question: Question,
        endpoint: ModelEndpoint | None,
        report: Callable[[Progress], None],
    ) -> Mapping[Any, Any]:
        """Work done once for many students' ``answers`` to ``question``, distinct and as they
        wrote them, that its ``checks`` mark, before they are marked. Marking finds it in
        ``Context.precomputed`` under the method's ref and the question."""
        return {}


class AnswerMethod[S: Settings](Method[S]):
    """A method that marks one answer: the question's, or one value of it."""

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        assert question is not None  # a check that marks an answer has a question
        answer = read_answer(context.answers, question, value)
        assert not isinstance(answer, Unreadable)  # marking deals with these first
        model = Position(context.questions, question, value).question_model
        assert model is not None  # problems refuse checks of questions not in the set
        return {question: self._mark_answer(self._settings_of(check), answer, model)}

    @abstractmethod
    def _mark_answer(self, settings: S, answer: Answer, question: Question) -> Outcome: ...
