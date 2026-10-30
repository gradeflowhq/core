from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from pydantic import BaseModel, ConfigDict, Field

from ..common import schemas
from ..common.hints import hint
from ..common.problems import as_invalid
from ..common.schemas import JsonDict
from .reading import write_answer
from .types import Answer, AnswerShape, QuestionType


class Question(BaseModel, ABC):
    """A question of one of the types, which reads answers as its type says."""

    label: ClassVar[str]
    summary: ClassVar[str]
    """One sentence."""
    starting: ClassVar[dict[str, Any]] = {}
    """What a new question of this type is set up with, beyond its defaults."""
    if TYPE_CHECKING:
        type: QuestionType  # each type's own field, which would come first here

    max_marks: float = Field(
        default=1.0,
        ge=0,
        allow_inf_nan=False,
        title="Maximum marks",
        description="The most marks a student can get for this question.",
    )
    description: str | None = Field(
        default=None,
        title="Description",
        description="A note about the question, such as what it asks, in Markdown. Formulas go "
        "between $ signs in LaTeX, like $x^2$.",
        json_schema_extra=hint(kind="markdown"),
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "label" in cls.__dict__:
            cls.model_config = ConfigDict(**{**cls.model_config, "title": f"{cls.label} question"})

    @classmethod
    def load(cls, data: object) -> Self:
        """The question ``data`` holds as YAML or JSON holds it: of the type its ``type`` names,
        or of this type, which it needn't name. Raises ``InvalidError`` if it isn't one."""
        from . import ANY_QUESTION  # the types are built on this class

        with as_invalid(data):
            if cls is Question:
                return cast(Self, ANY_QUESTION.validate_python(data))
            return cls.model_validate(data)

    def dump(self) -> dict[str, Any]:
        """The question as YAML or JSON holds it, as ``load`` reads it: with its type, without
        what it leaves at the defaults."""
        return {"type": self.type, **self.model_dump(mode="json", exclude_defaults=True)}

    @classmethod
    def json_schema(cls) -> JsonDict:
        """The JSON schema of a question of this type's settings."""
        return schemas.json_schema(cls)

    @property
    def value_count(self) -> int:
        """How many values an answer has: none, unless it has several."""
        return 0

    def value_question(self, value: int) -> Question | None:
        """A question reading answers as this one reads value ``value`` of each, from 1; None if
        they have no such value."""
        return None

    @property
    def answer_shape(self) -> AnswerShape:
        return "single"

    def single(self) -> Self:
        """This question reading each answer as one, for a check marking one of several."""
        return self

    def first(self, answer: Answer) -> Answer:
        """The likeliest of the possible answers ``answer`` is; the answer itself unless it
        reads answers as several."""
        return answer

    @abstractmethod
    def python_type(self) -> str:
        """The type custom code gets an answer as, in Python."""

    @abstractmethod
    def read(self, answer: str) -> Answer:
        """What ``answer`` reads as, or None if it is blank. Raises ``ValueError`` with the
        reason if it can't be read."""

    def normalize(self, answer: str) -> str:
        """
        ``answer`` as this question reads it, so answers it reads alike are written alike: ``42.0``
        and ``42``, or the choices ``B,D`` and ``D,B``. A blank answer is ``""``, and one the
        question can't read stays as it is.
        """
        try:
            return write_answer(self.first(self.read(answer)))
        except ValueError:
            return answer

    def answer_schema(self) -> JsonDict:
        """The JSON schema of an answer: text, for one written whole, or an object of the values
        it is made of."""
        return {"type": "string"}

    def answer_values(self, answer: str) -> dict[str, Any]:
        """``answer`` as the values of `answer_schema`, each as written."""
        raise ValueError(f"{self.label} answers are written whole.")

    def answer_from(self, values: Mapping[str, Any]) -> str:
        """The answer the values of `answer_schema` write; blank when they all are."""
        raise ValueError(f"{self.label} answers are written whole.")

    def accepting(self, answers: Sequence[str]) -> Self:
        """This question, changed to accept everything in ``answers``."""
        return self
