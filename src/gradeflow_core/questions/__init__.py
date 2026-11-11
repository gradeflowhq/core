"""Questions of each type and how they read answers, question sets, inferring questions from
answers, and answer sheets."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, get_args

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
QUESTION_ADAPTER: TypeAdapter[AnyQuestion] = TypeAdapter(AnyQuestion)
QUESTION_CLASSES: Mapping[QuestionType, type[Question]] = MappingProxyType(
    {cls.model_fields["type"].default: cls for cls in get_args(get_args(AnyQuestion)[0])}
)
"""Each type of question, in catalogue order: the first is the default for new questions."""


def new_question(question_type: QuestionType) -> Question:
    """A question of ``question_type`` to start from."""
    question_class = QUESTION_CLASSES[question_type]
    return question_class.load(question_class.new_question_data)
