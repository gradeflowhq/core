from ...questions.base import Question
from ...questions.types import ALL_TYPES, Answer
from ..check import Check
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import Outcome
from ..settings import Settings


class FullMarksSettings(Settings):
    pass


class FullMarks(AnswerMethod[FullMarksSettings]):
    name = "gradeflow/full-marks"
    version = 1
    label = "Full Marks"
    summary = "Full marks for every student."
    guide = (
        "**Use it when:** A question was faulty or dropped, and every student should get its "
        "marks.\n\n"
        "**How it marks:** Every student gets full marks, blank answers included.\n\n"
        "**Related:** To give marks only to students who answered, use Length with **At least** "
        "`1` character."
    )
    question_types = ALL_TYPES
    answer_shape = "both"
    marks_blank = True
    settings = FullMarksSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        return "Full marks for every answer."

    def _mark_answer(
        self, settings: FullMarksSettings, answer: Answer, question: Question
    ) -> Outcome:
        return Outcome(1.0, "Full marks.")
