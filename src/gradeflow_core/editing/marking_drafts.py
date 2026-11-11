"""A marking of a question or question group being edited."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

from ..common.documents import line_of, load_yaml_with_lines
from ..common.exceptions import EditingError
from ..common.problems import InvalidError, Problem
from ..methods.catalogue import BUILTINS
from ..methods.collection import Methods
from ..questions.base import Question
from ..questions.question_set import QuestionSet
from ..questions.types import QuestionId
from ..steps.step import Step, needs_workflow_problem
from .by_example import Decision, answer_decisions, mark_by_example
from .drafts import Draft


class MarkingDraft(Draft):
    """A marking of ``question``, or of a question group without one, as it is being edited;
    ``""`` is the marking itself."""

    @classmethod
    def from_yaml(
        cls,
        text: str,
        questions: QuestionSet,
        question: QuestionId | None,
        *,
        methods: Methods = BUILTINS,
    ) -> MarkingDraft:
        """The marking ``text`` writes of ``question``, which it needn't name. Raises
        ``InvalidError`` for text that isn't one check, or names another question."""
        document, data = load_yaml_with_lines(text)
        if not isinstance(data, dict):
            raise InvalidError(
                [Problem("Write one check, starting with uses: and its method's name.")]
            )
        names_other_question = "question" in data and data["question"] != question
        data = (
            {**data, "question": question}
            if question
            else {key: value for key, value in data.items() if key != "question"}
        )
        draft = cls(data, questions, question, methods=methods)
        draft._snapshot = replace(draft._snapshot, document=document)
        if names_other_question:
            fix_message = (
                f"Set question to {question}."
                if question
                else "Remove question; a question group's checks name their own."
            )
            question_problem = Problem(fix_message, line=line_of(document, "/question"))
            raise InvalidError(list(dict.fromkeys([question_problem, *draft.problems])))
        return draft

    @classmethod
    def new(
        cls,
        reference: str,
        questions: QuestionSet,
        question: QuestionId | None,
        *,
        methods: Methods = BUILTINS,
    ) -> MarkingDraft:
        """A draft of new marking with the method ``reference`` names. Raises ``EditingError`` if it
        can't mark it."""
        draft = cls({}, questions, question, methods=methods)
        draft.replace(draft._new_step_at("", draft._method(reference), None, names_question=True))
        return draft

    def answer_decisions(self) -> dict[str, Decision]:
        """The answers given marks by example, with what each gets.

        Keys use the question's spelling. Other methods' matches are not decisions. A decision
        describes the explicit rule, independently of conditions and other methods' results.
        """
        return answer_decisions(self._data, self._methods, self._question)

    def mark_by_example(
        self,
        answers: Sequence[str],
        decision: Decision | Literal["cancel"],
        proportion: float = 0.5,
    ) -> None:
        """Gives these answers one of the ``DECISIONS``, ``proportion`` of the marks for
        ``"partial_marks"``, or takes their decision back with ``"cancel"``. Other answers keep
        theirs.

        A decision overrides the existing marking. Cancelling leaves the remaining methods to mark
        the answers; an answer must be cancelled before it gets another decision, and giving one
        the decision it has changes nothing. Equivalent spellings of the question's answers are
        one answer; its type must be one that can be marked by example. New checks use the
        highest version of each method.
        """
        marking = mark_by_example(
            self._data, answers, decision, proportion, self._methods, self._question
        )
        if marking is not None:
            self.replace(marking)

    @property
    def _question(self) -> Question | None:
        return self.questions[self.question] if self.question else None

    def finalized(self) -> Step:
        """The marking, once nothing is wrong with it. Raises ``EditingError`` with its first
        problem."""
        marking, problems = self._snapshot.marking_places[""].step, self._marking_problems()
        if problems or marking is None:
            raise EditingError(str(problems[0]))
        return marking

    @property
    def step_problems(self) -> list[Problem]:
        """What is wrong with each step, the marking itself included, at the step and field at
        fault; as read from YAML, each on its line too, and one of those reading alike."""
        problems, document = self._marking_problems(), self._snapshot.document
        if document is None:
            return problems
        problems_by_text: dict[str, Problem] = {}
        for problem in problems:
            line = line_of(
                document, problem.where if problem.field else problem.step_path + "/uses"
            )
            problem_with_line = replace(problem, line=line)
            problems_by_text.setdefault(str(problem_with_line), problem_with_line)
        return list(problems_by_text.values())

    def _marking_problems(self) -> list[Problem]:
        """What is wrong with its steps; first, a marking that gives no marks."""
        marking = self._snapshot.marking_places[""].step
        if (
            marking is not None
            and not marking.resolution.gives_marks
            and not marking.resolution.data_problems
        ):
            return [needs_workflow_problem(marking)]
        return super().step_problems
