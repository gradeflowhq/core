"""An ExamSoft answer key row, read as answers to set questions up from."""

from collections.abc import Mapping

from ..questions.submissions import AnswerSheet
from ..questions.types import QuestionId
from ..questions.values import BLANK_SEPARATOR


def read_answer_key(row: Mapping[QuestionId, str]) -> AnswerSheet:
    """The answers an answer key row gives, by question, to set questions up from with the
    students' answers. A cell's blanks are separated by ``~`` and a blank's accepted answers by
    ``|``; each blank's first accepted answer stands for it, so ``0.63|0.625|~Terminal|terminal|``
    is ``0.63~Terminal``. Empty cells answer nothing."""
    answers = {
        key: BLANK_SEPARATOR.join(_first(blank) for blank in cell.split(BLANK_SEPARATOR))
        for key, cell in row.items()
        if cell.strip()
    }
    return AnswerSheet(answers=answers)


def _first(blank: str) -> str:
    return next((accepted for accepted in blank.split("|") if accepted.strip()), "")
