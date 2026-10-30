"""A mark scheme: an assessment's markings, each a check of a question or a question group."""

from typing import Any

from pydantic import BaseModel, Field

from ..common.documents import dump_yaml, load_yaml
from ..common.exceptions import InvalidError
from ..common.paths import join
from ..common.problems import Problem, as_invalid
from ..methods.catalogue import BUILTINS
from ..methods.check import METHODS, Marking
from ..methods.collection import Methods
from ..methods.problems import located_problems
from ..methods.templates import TemplateError
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId


class MarkScheme(BaseModel):
    markings: list[Marking] = Field(default_factory=list)

    @classmethod
    def load(cls, data: object, *, methods: Methods = BUILTINS) -> MarkScheme:
        """The markings ``data`` lists as a mark scheme whose checks may use ``methods``. Raises
        ``InvalidError`` if one isn't a check."""
        scheme = {"markings": data}
        with as_invalid(scheme):
            return cls.model_validate(scheme, context={METHODS: methods.by_ref})

    @classmethod
    def from_yaml(cls, text: str, *, methods: Methods = BUILTINS) -> MarkScheme:
        """The mark scheme YAML ``text`` holds, as ``to_yaml`` writes it. Raises ``InvalidError``
        if it isn't YAML or a marking isn't a check."""
        return cls.load(load_yaml(text), methods=methods)

    def dump(self) -> list[dict[str, Any]]:
        """The markings as YAML or JSON holds them, as ``load`` reads them."""
        return [marking.dump() for marking in self.markings]

    def to_yaml(self) -> str:
        """The markings as YAML text, as ``from_yaml`` reads them."""
        return dump_yaml(self.dump())

    def needs_model_endpoint(self) -> bool:
        """Whether marking asks a language model, so ``mark`` needs a model endpoint."""
        return any(marking.needs_model_endpoint() for marking in self.markings)

    def problems(self, questions: QuestionSet) -> list[Problem]:
        found = [
            problem
            for number, marking in enumerate(self.markings)
            for problem in located_problems(marking, questions, ("markings", number))
        ]
        first: dict[QuestionId, int] = {}
        for number, marking in enumerate(self.markings):
            try:
                outputs = marking.questions_marked()
            except InvalidError, TemplateError:
                continue  # Its generated body already has a validation problem above.
            found += [
                Problem(f"{key} is marked twice.", join("markings", number))
                for key in sorted(outputs)
                if first.setdefault(key, number) != number
            ]
        return found
