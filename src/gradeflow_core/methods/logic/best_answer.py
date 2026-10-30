from dataclasses import replace
from typing import Annotated, Literal

from pydantic import Field

from ...common.hints import hint
from ...common.paths import Loc
from ...questions.question_set import QuestionSet
from ...questions.reading import write_answer
from ...questions.types import ALL_TYPES, QuestionId, Unreadable
from ..check import Check, Context, Marking
from ..example import examples_beside
from ..method import Method
from ..outcome import Outcome
from ..position import Position
from ..settings import Settings

Keep = Literal["most_marks"]
KEEP_LABELS = {"most_marks": "Most marks"}


class BestAnswerSettings(Settings):
    check: Annotated[
        Marking,
        Field(
            title="Check",
            description="The check for each possible answer.",
            json_schema_extra=hint(kind="check"),
        ),
    ]
    keep: Keep = Field(
        default="most_marks",
        title="Keep",
        json_schema_extra=hint(labels=KEEP_LABELS, advanced=True),
    )


class BestAnswer(Method[BestAnswerSettings]):
    name = "gradeflow/best-answer"
    version = 1
    label = "Best Answer"
    summary = "Marks for the possible answer that scores highest."
    guide = (
        "**Use it when:** The question reads each answer as several possible answers, such as "
        "code read with **Guess lost indentation**.\n\n"
        "**How it marks:** The check inside marks each possible answer, and the one with the "
        "most marks counts. On a tie, the likeliest counts.\n\n"
        "**Good to know:** When the answer that counts isn't the likeliest, the feedback shows it "
        "first, so students see what was marked."
    )
    question_types = ALL_TYPES
    answer_shape = "multiple"
    settings = BestAnswerSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        return f"**Best possible answer:** {self._settings_of(check).check.describe()}"

    def inside(self, position: Position, loc: Loc) -> Position:
        return replace(position, one_answer=True)

    def mark(
        self,
        check: Check,
        context: Context,
        question: QuestionId | None,
        value: int | None,
    ) -> dict[QuestionId, Outcome]:
        assert question is not None  # it marks a question read as several answers
        inner = self._settings_of(check).check
        read = context.answers.get(question)
        assert not isinstance(read, Unreadable)  # marking deals with these first
        possible = read if isinstance(read, list) else [read]
        questions = QuestionSet(
            {**context.questions, question: context.questions[question].single()}
        )
        marked = [
            inner.mark(
                replace(
                    context, answers={**context.answers, question: answer}, questions=questions
                ),
                question,
            )
            for answer in possible
        ]
        best = max(range(len(marked)), key=lambda index: _score(marked[index].get(question)))
        outcomes = marked[best]
        if best == 0 or question not in outcomes:
            return outcomes
        kept = outcomes[question]
        feedback = f"Read as:\n{write_answer(possible[best])}\n\n{kept.feedback}".rstrip()
        return {**outcomes, question: replace(kept, feedback=feedback)}

    def questions_marked(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return self._settings_of(check).check.questions_marked(question)

    def questions_read(self, check: Check, question: QuestionId | None) -> set[QuestionId]:
        return self._settings_of(check).check.questions_read(question)


def _score(outcome: Outcome | None) -> tuple[bool, float]:
    """Marked outcomes first, then by marks; ``max`` keeps the earliest of equals."""
    return (False, 0.0) if outcome is None else (outcome.marked, outcome.proportion)
