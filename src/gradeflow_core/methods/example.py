"""A method's examples: questions, settings and answers, with the results declared for them."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..common.documents import load_yaml
from ..questions import AnyQuestion
from ..questions.submissions import AnswerSheet
from ..questions.types import QuestionId


class ExpectedResult(BaseModel):
    """The marks and feedback one answer gets."""

    model_config = ConfigDict(extra="forbid")

    marks: float
    feedback: str


class ExampleRow(AnswerSheet):
    """An answer sheet, and the result declared for each question it is marked for, if any."""

    model_config = ConfigDict(extra="forbid")

    expected: dict[QuestionId, ExpectedResult] = Field(default_factory=dict)


class Example(BaseModel):
    """A self-contained example, never executed when read."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    title: str
    description: str
    questions: dict[QuestionId, AnyQuestion]
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")
    rows: list[ExampleRow]


def examples_beside(path: str) -> tuple[Example, ...]:
    """The examples of the method written in the module at ``path``, from the YAML file beside it
    named as the method is: ``accepted-answers.examples.yaml`` beside ``accepted_answers.py``."""
    module = Path(path)
    text = module.with_name(f"{module.stem.replace('_', '-')}.examples.yaml").read_text("utf-8")
    return tuple(Example.model_validate(data) for data in load_yaml(text)["examples"])
