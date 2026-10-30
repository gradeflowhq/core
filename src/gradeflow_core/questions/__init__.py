from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, get_args

from pydantic import Field, TypeAdapter

from .base import Question
from .choice import ChoiceQuestion
from .code import CodeQuestion
from .number import NumberQuestion
from .text import TextQuestion
from .types import QuestionType
from .values import ValuesQuestion

if TYPE_CHECKING:
    # Every question is a Question, so a dict of them builds a QuestionSet without a cast.
    AnyQuestion = Question
else:
    # Catalogue order; the first is the default for new questions.
    AnyQuestion = Annotated[
        TextQuestion | NumberQuestion | ChoiceQuestion | ValuesQuestion | CodeQuestion,
        Field(discriminator="type"),
    ]
    """A question of any type, validated as the type its ``type`` names."""
ANY_QUESTION: TypeAdapter[AnyQuestion] = TypeAdapter(AnyQuestion)
QUESTION_TYPES: Mapping[QuestionType, type[Question]] = MappingProxyType(
    {cls.model_fields["type"].default: cls for cls in get_args(get_args(AnyQuestion)[0])}
)
"""Each type of question, in catalogue order: the first is the default for new questions."""
TYPE_LABELS = {kind: cls.label for kind, cls in QUESTION_TYPES.items()}


def new_question(question_type: QuestionType, data: Mapping[str, Any] | None = None) -> Question:
    """A question of ``question_type`` from ``data``, as its JSON schema describes it; without it,
    one to start from. Raises ``InvalidError`` if it isn't valid."""
    kind = QUESTION_TYPES[question_type]
    return kind.load({**(kind.starting if data is None else data), "type": question_type})
