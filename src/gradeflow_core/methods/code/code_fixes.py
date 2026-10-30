from pydantic import Field

from ...common.text import capitalised, format_number, plural
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..method import AnswerMethod
from ..outcome import Outcome
from ..settings import Settings
from .code_test import FIXES, PENALTY, TestSetup
from .fixes import fixed_code


class CodeFixesSettings(Settings):
    setup: TestSetup = Field(default_factory=TestSetup, title="Test setup")
    fixes: int = FIXES
    penalty: float = PENALTY


class CodeFixes(AnswerMethod[CodeFixesSettings]):
    name = "gradeflow/code-fixes"
    version = 1
    label = "Code Fixes"
    summary = "Full marks if small errors can be fixed."
    level = "internal"
    question_types = frozenset({"code"})
    settings = CodeFixesSettings

    def describe(self, check: Check) -> str:
        return "Has small errors that can be fixed."

    def _mark_answer(
        self, settings: CodeFixesSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(answer, str)  # a code question reading one answer
        fixed = fixed_code(answer, settings.setup.around("pass"), settings.fixes)
        if fixed is None:
            return Outcome(0.0)
        count = len(fixed.fixes)
        fixes = " ".join(f"{capitalised(fix)}." for fix in fixed.fixes)
        off = f", for {format_number(settings.penalty * 100)}% off" if settings.penalty else ""
        return Outcome(
            1.0,
            f"Fixed {count} small {plural('error', count)} before testing{off}. {fixes}",
        )
