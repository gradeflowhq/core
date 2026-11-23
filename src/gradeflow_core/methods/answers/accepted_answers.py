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
from ...models import nli
from ...questions.base import Question
from ...questions.types import Answer
from ...steps.example import examples_beside
from ...steps.method import AnswerMethod
from ...steps.outcome import CORRECT_FEEDBACK, Outcome
from ...steps.position import Position
from ...steps.settings import NonBlankText, Number, Settings, specialized
from ..matching import (
    MATCH_FIELD,
    SPELLING_CUTOFFS,
    STRICTNESS_FIELD,
    TOLERANCE_FIELD,
    MatchMode,
    Strictness,
    closest_spelling_index,
    expected_text,
    is_within_tolerance,
    match_note,
    model_context,
    unreadable_problems,
)

# The lower, over both directions, of entailment minus contradiction; calibrated on real students'
# answers to science, civics and computing questions.
ANSWER_MEANING_CUTOFFS: dict[Strictness, float] = {"lenient": -0.1, "normal": 0.005, "strict": 0.5}


class AcceptedAnswersSettings(Settings):
    answers: list[NonBlankText] = Field(
        title="Accepted answers",
        json_schema_extra=hint(lists_answers=True),
    )
    match: MatchMode = MATCH_FIELD
    strictness: Strictness = STRICTNESS_FIELD
    tolerance: Number = TOLERANCE_FIELD


class AcceptedAnswers(AnswerMethod[AcceptedAnswersSettings]):
    name = "gradeflow/accepted-answers"
    version = 1
    label = "Accepted Answers"
    summary = "Correct if the answer matches one you accept."
    guide = (
        "For a question with one or a few right answers: a name, a term, a number or a short "
        "sentence.\n\n"
        "The whole answer is compared with each accepted answer, and it is correct if it matches "
        "any of them. **Match** sets how close it must be:\n\n"
        "- **Exactly**: the same text. Capitals don't count while the question's **Ignore case** "
        "is on, as it is by default. Punctuation counts unless **Ignore punctuation** is on.\n"
        "- **Similar spelling**: the same text with small typos, so “mitocondria” is accepted for "
        "“mitochondria”. **How strict** sets how many typos pass.\n"
        "- **Same meaning**: the same statement in other words, so “Its density is lower than that "
        "of water” is accepted for “Ice is less dense than liquid water”.\n\n"
        "Same meaning checks both ways: the answer must say what the accepted answer says, and "
        "nothing more. With **How strict** at **Normal**, “Ice is less dense than water because "
        "water expands when it freezes” isn't accepted, though it's right. **Lenient** accepts "
        "answers that add to yours, and **Strict** accepts only close rewordings. To accept every "
        "answer that makes a point, whatever else it says, use Key Points with Same meaning.\n\n"
        "Same meaning reads the question's **Description** with each answer, so write the question "
        "there: “It's less dense” is then accepted. A language model judges within GradeFlow, and "
        "answers aren't sent to your AI service. Preview the marks to see what it accepts.\n\n"
        "On a number question, **Tolerance** accepts numbers near an accepted one. With `9.81` and "
        "a tolerance of `0.05`, any answer from `9.76` to `9.86` is correct. Number answers may "
        "have units and powers of ten, so `9.81 m/s^2` reads as 9.81.\n\n"
        "Exactly and Similar spelling count extra words against an answer, so “the mitochondria” "
        "doesn't match “mitochondria”. Add it as another accepted answer, or use Key Points. "
        "Similar spelling can also accept a different word that looks alike, such as “ilium” for "
        "“ileum”.\n\n"
        "To give part marks for a nearly right answer, put this check in a Workflow with **Any one "
        "must be correct** and add a second Accepted Answers check with a lower weight. To mark "
        "the points an answer makes, use Key Points."
    )
    question_types = frozenset({"text", "number", "code"})
    settings = AcceptedAnswersSettings
    examples = examples_beside(__file__)

    def settings_model_at(self, position: Position) -> type[Settings]:
        hidden_fields = {
            "number": ("match", "strictness"),
            "code": ("match", "strictness", "tolerance"),
        }.get(position.question_type or "", ("tolerance",))
        return specialized(AcceptedAnswersSettings, excluded_fields=hidden_fields)

    def describe(self, settings: AcceptedAnswersSettings) -> str:
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

    def mark_answer(
        self, settings: AcceptedAnswersSettings, answer: Answer, question: Question
    ) -> Outcome:
        if not settings.answers:
            return Outcome(0.0, "No accepted answers.")
        accepted_values = [question.read(accepted_answer) for accepted_answer in settings.answers]
        if isinstance(answer, int | float):
            within_tolerance = any(
                is_within_tolerance(answer, value, settings.tolerance)
                for value in accepted_values
                if isinstance(value, int | float)
            )
            if within_tolerance:
                return Outcome(1.0, CORRECT_FEEDBACK)
            accepted_answers_phrase = plain_join(settings.answers, conjunction="or")
            return Outcome(0.0, f"{expected_text(accepted_answers_phrase, settings.tolerance)}.")
        answer_text = str(answer)
        if answer_text in accepted_values:
            return Outcome(1.0, CORRECT_FEEDBACK)
        not_accepted_outcome = Outcome(
            0.0, as_sentence(f"Expected {quoted_join(settings.answers, conjunction='or')}")
        )
        if settings.match == "exact":
            return not_accepted_outcome
        accepted_texts = list(map(str, accepted_values))
        if settings.match == "spelling":
            index = closest_spelling_index(
                answer_text, accepted_texts, SPELLING_CUTOFFS[settings.strictness]
            )
        else:
            cutoff = ANSWER_MEANING_CUTOFFS[settings.strictness]
            index = nli.first_match_index(
                answer_text, accepted_texts, model_context(question), cutoff
            )
        if index is None:
            return not_accepted_outcome
        return Outcome(1.0, f"Accepted as {quoted(settings.answers[index])}.")

    def problems(self, settings: AcceptedAnswersSettings, position: Position) -> Iterator[Problem]:
        question = position.question_model
        if question is not None:
            yield from unreadable_problems(settings.answers, question.read, "answers")
