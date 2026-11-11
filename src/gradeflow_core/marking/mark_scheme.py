"""A mark scheme: an assessment's markings, each a check of a question or a question group."""

from dataclasses import dataclass, field, replace
from typing import Any

from ..common.documents import dump_yaml, load_yaml
from ..common.paths import pointer
from ..common.problems import InvalidError, Problem
from ..methods.catalogue import BUILTINS
from ..methods.collection import Methods
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from ..steps.places import marking_position
from ..steps.problems import located_problems
from ..steps.step import Step, needs_workflow_problem, valid_step


@dataclass(frozen=True)
class MarkScheme:
    markings: list[Step] = field(default_factory=list)

    @classmethod
    def load(cls, data: object, *, methods: Methods = BUILTINS) -> MarkScheme:
        """The markings ``data`` lists as a mark scheme whose checks may use ``methods``. Raises
        ``InvalidError`` if one isn't a check."""
        if not isinstance(data, list):
            raise InvalidError([Problem("Must be a list.", field="/markings")])
        markings: list[Step] = []
        problems: list[Problem] = []
        for index, marking_data in enumerate(data):
            try:
                marking = valid_step(marking_data, methods.by_ref)
            except InvalidError as error:
                problems += [
                    replace(problem, step_path=pointer("markings", index) + problem.step_path)
                    for problem in error.problems
                ]
                continue
            if not marking.resolution.gives_marks:
                problem = needs_workflow_problem(marking)
                problems.append(replace(problem, step_path=pointer("markings", index)))
            markings.append(marking)
        if problems:
            raise InvalidError(problems)
        return cls(markings)

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
        problems = [
            problem
            for index, marking in enumerate(self.markings)
            for problem in located_problems(
                marking, marking_position(questions, marking.question), pointer("markings", index)
            )
        ]
        first_marking_index_by_question: dict[QuestionId, int] = {}
        for index, marking in enumerate(self.markings):
            problems += [
                Problem(f"{question_id} is marked twice.", pointer("markings", index))
                for question_id in sorted(marking.questions_marked())
                if first_marking_index_by_question.setdefault(question_id, index) != index
            ]
        return problems
