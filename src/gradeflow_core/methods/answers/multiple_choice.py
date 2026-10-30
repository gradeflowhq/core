from collections.abc import Iterator, Mapping
from typing import Any, Literal

from pydantic import Field, field_validator

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import markdown_join, quoted_join, sentence
from ...questions.base import Question
from ...questions.choice import ChoiceQuestion
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import CORRECT, Outcome
from ..position import Position
from ..settings import Settings, literal, specialised

Compare = Literal["exactly", "all", "none", "any", "each"]
COMPARE_LABELS = {
    "exactly": "Exactly these",
    "all": "All of these, extras allowed",
    "none": "None of these",
    "any": "Any one of these",
    "each": "Each one counts",
}


class MultipleChoiceSettings(Settings):
    correct: list[str] = Field(
        min_length=1,
        title="Correct options",
        json_schema_extra={"uniqueItems": True},
    )
    compare: Compare = Field(
        default="exactly",
        title="How to compare",
        description="Each one counts gives a share for each right option and takes one off for "
        "each wrong one.",
        json_schema_extra=hint(labels=COMPARE_LABELS),
    )

    @field_validator("correct")
    @classmethod
    def _once_each(cls, correct: list[str]) -> list[str]:
        return list(dict.fromkeys(correct))


class MultipleChoice(AnswerMethod[MultipleChoiceSettings]):
    name = "gradeflow/multiple-choice"
    version = 1
    label = "Multiple Choice"
    summary = "Marks for choosing the right options."
    guide = (
        "**Use it when:** Students choose one or more options.\n\n"
        "**How it marks:** The options chosen are compared with the right ones, as **How to "
        "compare** says. **Exactly these** needs the right options and nothing else. **Each one "
        "counts** gives a share for each right option and takes one off for each wrong one, never "
        "below zero."
    )
    question_types = frozenset({"choice"})
    settings = MultipleChoiceSettings
    examples = examples_beside(__file__)

    def settings_at(self, position: Position) -> type[Settings]:
        question = position.question_model
        if not isinstance(question, ChoiceQuestion):
            return self.settings
        options = literal(sorted(question.options))
        return specialised(self.settings, correct={"annotation": list[options]})  # type: ignore[valid-type]

    def read_at(self, settings: Mapping[str, Any], position: Position) -> dict[str, Any]:
        question, correct = position.question_model, settings.get("correct")
        if not isinstance(question, ChoiceQuestion) or not isinstance(correct, list):
            return dict(settings)
        read = (
            question.reading.option(each) if isinstance(each, str) else each for each in correct
        )
        return {**settings, "correct": list(dict.fromkeys(read))}

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        either = markdown_join(settings.correct, conjunction="or")
        every = markdown_join(settings.correct, conjunction="and")
        match settings.compare:
            case "exactly":
                return f"Chooses exactly {every}."
            case "all":
                return f"Chooses {every}, extras allowed."
            case "none":
                return f"Chooses none of {every}."
            case "any":
                return f"Chooses {either}."
            case "each":
                return f"Chooses {every}, each counting."

    def _mark_answer(
        self, settings: MultipleChoiceSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(question, ChoiceQuestion)
        chosen = answer if isinstance(answer, frozenset) else frozenset()
        right = {question.reading.option(option) for option in settings.correct}
        match settings.compare:
            case "exactly":
                proportion = float(chosen == right)
            case "all":
                proportion = float(right <= chosen)
            case "none":
                proportion = float(not right & chosen)
            case "any":
                proportion = float(bool(right & chosen))
            case "each":
                proportion = max(0.0, (len(chosen & right) - len(chosen - right)) / len(right))
        if proportion >= 1:
            return Outcome(1.0, CORRECT)
        listed = quoted_join(
            settings.correct, conjunction="or" if settings.compare == "any" else "and"
        )
        none = "none of " if settings.compare == "none" else ""
        return Outcome(proportion, sentence(f"Expected {none}{listed}"))

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        question = position.question_model
        if not isinstance(question, ChoiceQuestion):
            return
        for index, option in enumerate(self._settings_of(check).correct):
            if question.reading.option(option) not in question.options:
                yield Problem(f"{option} isn't one of the options.", field=("correct", index))
