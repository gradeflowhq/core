from collections.abc import Iterator, Mapping
from typing import Any, Literal

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import (
    as_sentence,
    code_join,
    markdown_code_list,
    quoted_join,
    reads_better_listed,
)
from ...questions.base import Question
from ...questions.choice import ChoiceQuestion
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..matching import duplicate_problems
from ..method import AnswerMethod
from ..outcome import CORRECT, Outcome
from ..position import Position
from ..settings import Settings, literal, specialized

Compare = Literal["exactly", "all", "none", "any", "share_each"]
COMPARE_LABELS = {
    "exactly": "Exactly these",
    "all": "All of these, other options allowed",
    "none": "None of these",
    "any": "Any one of these",
    "share_each": "A share for each",
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
        description="A share for each gives a share for each correct option and a penalty of one "
        "share for each wrong one.",
        json_schema_extra=hint(labels=COMPARE_LABELS),
    )


_DESCRIPTIONS: dict[Compare, tuple[str, str, Literal["and", "or"]]] = {
    "exactly": ("Chooses exactly {}.", "Chooses exactly these", "and"),
    "all": (
        "Chooses {}, other options allowed.",
        "Chooses all of these, other options allowed",
        "and",
    ),
    "none": ("Chooses none of {}.", "Chooses none of these", "and"),
    "any": ("Chooses {}.", "Chooses any of these", "or"),
    "share_each": ("Chooses {}, a share for each.", "Chooses these, a share for each", "and"),
}
"""For each way of comparing: its description with the options inline, how one listing them
opens, and the word that joins them."""


class MultipleChoice(AnswerMethod[MultipleChoiceSettings]):
    name = "gradeflow/multiple-choice"
    version = 1
    label = "Multiple Choice"
    summary = "Marks for choosing the correct options."
    guide = (
        "**Use it when:** Students choose one or more options.\n\n"
        "**How it marks:** The options chosen are compared with the correct ones, as **How to "
        "compare** says. **Exactly these** needs the correct options and nothing else. **A share "
        "for each** gives a share for each correct option and a penalty of one share for each "
        "wrong one, never below zero."
    )
    question_types = frozenset({"choice"})
    settings = MultipleChoiceSettings
    examples = examples_beside(__file__)

    def settings_model_at(self, position: Position) -> type[Settings]:
        question = position.question_model
        if not isinstance(question, ChoiceQuestion):
            return self.settings
        options = literal(question.options)
        return specialized(self.settings, correct={"annotation": list[options]})  # type: ignore[valid-type]

    def settings_read_at(self, settings: Mapping[str, Any], position: Position) -> dict[str, Any]:
        question, correct = position.question_model, settings.get("correct")
        if not isinstance(question, ChoiceQuestion) or not isinstance(correct, list):
            return dict(settings)
        normalized_options = (
            question.reading.normalized(each) if isinstance(each, str) else each for each in correct
        )
        return {**settings, "correct": list(normalized_options)}

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        inline, listed, conjunction = _DESCRIPTIONS[settings.compare]
        if reads_better_listed(settings.correct):
            return f"{listed}:\n{markdown_code_list(settings.correct)}"
        return inline.format(code_join(settings.correct, conjunction=conjunction))

    def _mark_answer(
        self, settings: MultipleChoiceSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(question, ChoiceQuestion)
        assert isinstance(answer, frozenset)  # choice questions read sets of options
        chosen_options = answer
        correct_options = {question.reading.normalized(option) for option in settings.correct}
        match settings.compare:
            case "exactly":
                proportion = float(chosen_options == correct_options)
            case "all":
                proportion = float(correct_options <= chosen_options)
            case "none":
                proportion = float(not correct_options & chosen_options)
            case "any":
                proportion = float(bool(correct_options & chosen_options))
            case "share_each":
                chosen_correct = len(chosen_options & correct_options)
                chosen_wrong = len(chosen_options - correct_options)
                proportion = max(0.0, (chosen_correct - chosen_wrong) / len(correct_options))
        if proportion >= 1:
            return Outcome(1.0, CORRECT)
        expected_options = quoted_join(
            settings.correct, conjunction=_DESCRIPTIONS[settings.compare][2]
        )
        none_of_prefix = "none of " if settings.compare == "none" else ""
        return Outcome(proportion, as_sentence(f"Expected {none_of_prefix}{expected_options}"))

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        question = position.question_model
        if not isinstance(question, ChoiceQuestion):
            return
        correct = self._settings_of(check).correct
        for index, option in enumerate(correct):
            if question.reading.normalized(option) not in question.options:
                yield Problem(f"{option} isn't one of the options.", field=("correct", index))
        yield from duplicate_problems(correct, question.reading.normalized, "correct")
