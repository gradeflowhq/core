from collections.abc import Iterator

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import markdown_code, markdown_join, plain_join, quoted, quoted_join, sentence
from ...ml import nli
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..matching import (
    MATCH,
    SPELLING_CUTOFFS,
    TOLERANCE,
    Match,
    Strictness,
    context_of,
    expecting,
    how_matched,
    match_problems,
    most_alike,
    read_or_none,
    reading_problems,
    strictness_field,
    within,
)
from ..method import AnswerMethod
from ..outcome import CORRECT, Outcome
from ..position import Position
from ..settings import Settings, Text, specialised

# The lower, over both directions, of entailment minus contradiction; calibrated on real students'
# answers to science, civics and computing questions.
MEANING_CUTOFFS: dict[Strictness, float] = {"lenient": -0.1, "normal": 0.005, "strict": 0.5}


class AcceptedAnswersSettings(Settings):
    answers: list[Text] = Field(
        title="Accepted answers",
        json_schema_extra=hint(accepts_answers=True),
    )
    match: Match = MATCH
    strictness: Strictness = strictness_field(applies_when={"match": ["spelling", "meaning"]})
    tolerance: float = TOLERANCE


class AcceptedAnswers(AnswerMethod[AcceptedAnswersSettings]):
    name = "gradeflow/accepted-answers"
    version = 1
    label = "Accepted Answers"
    summary = "Full marks if the answer is one you accept."
    guide = (
        "**Use it when:** There are one or a few right answers, such as a name, a term, a number "
        "or a one-sentence explanation.\n\n"
        "**How it marks:** The whole answer is compared with each accepted answer, and a match "
        "gets full marks. **Exactly** needs the same text, and **Similar spelling** forgives "
        "small mistakes. **Same meaning** accepts other words, judged by a language model that "
        "also reads the question's description, and for number questions a **Tolerance** "
        "accepts nearby values.\n\n"
        "**Good to know:** Exactly ignores capitals but not punctuation, which the question's "
        "**Reading answers** can change. Same meaning is reliable only for a sentence or two, "
        "and Similar spelling can accept a look-alike word, such as “ilium” for “ileum”, so "
        "check which answers it accepts.\n\n"
        "**Related:** To mark the points an answer includes, use Key Points."
    )
    question_types = frozenset({"text", "number", "code"})
    settings = AcceptedAnswersSettings
    examples = examples_beside(__file__)

    def settings_at(self, position: Position) -> type[Settings]:
        drop = {
            "number": ("match", "strictness"),
            "code": ("match", "strictness", "tolerance"),
        }.get(position.answer_type or "", ("tolerance",))
        return specialised(AcceptedAnswersSettings, drop=drop)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        if not settings.answers:
            return "No accepted answers."
        answers = markdown_join(settings.answers, conjunction="or")
        if settings.tolerance > 0:
            return f"Matches {answers}, within {markdown_code(settings.tolerance)}."
        return sentence(f"Matches {answers}{how_matched(settings.match, settings.strictness)}")

    def _mark_answer(
        self, settings: AcceptedAnswersSettings, answer: Answer, question: Question
    ) -> Outcome:
        if not settings.answers:
            return Outcome(0.0, "No accepted answers.")
        accepted = {
            text: read
            for text in settings.answers
            if (read := read_or_none(question.read, text)) is not None
        }
        if isinstance(answer, int | float) and not isinstance(answer, bool):
            close = any(
                within(answer, value, settings.tolerance)
                for value in accepted.values()
                if isinstance(value, int | float)
            )
            if close:
                return Outcome(1.0, CORRECT)
            answers = plain_join(settings.answers, conjunction="or")
            return Outcome(0.0, f"{expecting(answers, settings.tolerance)}.")
        text = str(answer)
        if text in accepted.values():
            return Outcome(1.0, CORRECT)
        expected = Outcome(
            0.0, sentence(f"Expected {quoted_join(settings.answers, conjunction='or')}")
        )
        if settings.match == "exact" or not accepted:
            return expected
        written = list(accepted)
        references = [str(value) for value in accepted.values()]
        if settings.match == "spelling":
            index = most_alike(text, references, SPELLING_CUTOFFS[settings.strictness])
        else:
            cutoff = MEANING_CUTOFFS[settings.strictness]
            best, close = nli.best_match(text, references, context_of(question), cutoff)
            index = best if close else None
        if index is None:
            return expected
        return Outcome(1.0, f"Accepted as {quoted(written[index])}.")

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        yield from match_problems(settings.match)
        question = position.question_model
        if question is not None:
            yield from reading_problems(settings.answers, question.read, "answers")
