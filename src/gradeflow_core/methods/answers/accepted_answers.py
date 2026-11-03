from collections.abc import Iterator

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import (
    as_sentence,
    code_join,
    code_span,
    markdown_code_list,
    plain_join,
    quoted,
    quoted_join,
    reads_better_listed,
)
from ...ml import nli
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..matching import (
    MATCH_FIELD,
    SPELLING_CUTOFFS,
    TOLERANCE_FIELD,
    Match,
    Strictness,
    closest_spelling,
    expected_text,
    match_note,
    match_problems,
    model_context,
    read_or_none,
    strictness_field,
    unreadable_problems,
    within,
)
from ..method import AnswerMethod
from ..outcome import CORRECT, Outcome
from ..position import Position
from ..settings import NonBlankText, Settings, specialized

# The lower, over both directions, of entailment minus contradiction; calibrated on real students'
# answers to science, civics and computing questions.
ANSWER_MEANING_CUTOFFS: dict[Strictness, float] = {"lenient": -0.1, "normal": 0.005, "strict": 0.5}


class AcceptedAnswersSettings(Settings):
    answers: list[NonBlankText] = Field(
        title="Accepted answers",
        json_schema_extra=hint(lists_answers=True),
    )
    match: Match = MATCH_FIELD
    strictness: Strictness = strictness_field(applies_when={"match": ["spelling", "meaning"]})
    tolerance: float = TOLERANCE_FIELD


class AcceptedAnswers(AnswerMethod[AcceptedAnswersSettings]):
    name = "gradeflow/accepted-answers"
    version = 1
    label = "Accepted Answers"
    summary = "Full marks if the answer is one you accept."
    guide = (
        "**Use it when:** There are one or a few correct answers, such as a name, a term, a number "
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

    def settings_model_at(self, position: Position) -> type[Settings]:
        hidden_fields = {
            "number": ("match", "strictness"),
            "code": ("match", "strictness", "tolerance"),
        }.get(position.question_type or "", ("tolerance",))
        return specialized(AcceptedAnswersSettings, without=hidden_fields)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        if not settings.answers:
            return "No accepted answers."
        comparison_note = (
            f", within {code_span(settings.tolerance)}"
            if settings.tolerance > 0
            else match_note(settings.match, settings.strictness)
        )
        if reads_better_listed(settings.answers):
            return f"Matches any of these{comparison_note}:\n{markdown_code_list(settings.answers)}"
        return as_sentence(
            f"Matches {code_join(settings.answers, conjunction='or')}{comparison_note}"
        )

    def _mark_answer(
        self, settings: AcceptedAnswersSettings, answer: Answer, question: Question
    ) -> Outcome:
        if not settings.answers:
            return Outcome(0.0, "No accepted answers.")
        accepted = {
            text: value_read
            for text in settings.answers
            if (value_read := read_or_none(question.read, text)) is not None
        }
        if isinstance(answer, int | float) and not isinstance(answer, bool):
            within_tolerance = any(
                within(answer, value, settings.tolerance)
                for value in accepted.values()
                if isinstance(value, int | float)
            )
            if within_tolerance:
                return Outcome(1.0, CORRECT)
            answers = plain_join(settings.answers, conjunction="or")
            return Outcome(0.0, f"{expected_text(answers, settings.tolerance)}.")
        text = str(answer)
        if text in accepted.values():
            return Outcome(1.0, CORRECT)
        expected = Outcome(
            0.0, as_sentence(f"Expected {quoted_join(settings.answers, conjunction='or')}")
        )
        if settings.match == "exact" or not accepted:
            return expected
        accepted_texts = list(accepted)
        accepted_readings = [str(value) for value in accepted.values()]
        if settings.match == "spelling":
            index = closest_spelling(text, accepted_readings, SPELLING_CUTOFFS[settings.strictness])
        else:
            cutoff = ANSWER_MEANING_CUTOFFS[settings.strictness]
            best_index, is_match = nli.best_match(
                text, accepted_readings, model_context(question), cutoff
            )
            index = best_index if is_match else None
        if index is None:
            return expected
        return Outcome(1.0, f"Accepted as {quoted(accepted_texts[index])}.")

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        yield from match_problems(settings.match)
        question = position.question_model
        if question is not None:
            yield from unreadable_problems(settings.answers, question.read, "answers")
